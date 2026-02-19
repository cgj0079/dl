"""
MNIST 웹 추론 서버 - Flask
학습된 모델(best_model.pth)을 로드해서 실시간 손글씨 인식
"""

import io
import base64
import struct
import os
import math
import urllib.request
import gzip
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

app = Flask(__name__, static_folder='static')


# ─────────────────────────────────────────────
# 모델 정의 (mnist_cnn.py 와 동일 — ResNet + SE)
# ─────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, ch, ratio=4):
        super().__init__()
        mid = max(ch//ratio, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, mid), nn.GELU(),
            nn.Linear(mid, ch), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x).view(x.size(0),-1,1,1)

class ResBlock(nn.Module):
    def __init__(self, ch, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch,ch,3,padding=1,bias=False),
            nn.BatchNorm2d(ch), nn.GELU(),
            nn.Conv2d(ch,ch,3,padding=1,bias=False),
            nn.BatchNorm2d(ch))
        self.se  = SEBlock(ch)
        self.dp  = nn.Dropout2d(drop)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.se(self.dp(self.net(x))) + x)

class MNISTNet(nn.Module):
    def __init__(self, drop=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1,bias=False),
            nn.BatchNorm2d(32), nn.GELU())

        def stage(ci, co, n, dp):
            layers = [nn.Conv2d(ci,co,3,padding=1,bias=False),
                      nn.BatchNorm2d(co), nn.GELU()]
            layers += [ResBlock(co,dp) for _ in range(n)]
            return nn.Sequential(*layers)

        self.s1 = nn.Sequential(stage(32,  64, 2, 0.10), nn.MaxPool2d(2))
        self.s2 = nn.Sequential(stage(64, 128, 3, 0.15), nn.MaxPool2d(2))
        self.s3 = nn.Sequential(stage(128,256, 2, 0.20), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Dropout(drop), nn.Linear(256,10))

    def forward(self, x):
        return self.head(self.s3(self.s2(self.s1(self.stem(x)))))


# ─────────────────────────────────────────────
# 모델 로드
# ─────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')

model = MNISTNet(drop=0.3).to(DEVICE)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pth')

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    TRAINED = True
    print(f"[✓] 모델 로드 완료 - 디바이스: {DEVICE}")
    if 'val_acc' in checkpoint:
        print(f"[✓] 모델 검증 정확도: {checkpoint['val_acc']:.2f}%")
else:
    TRAINED = False
    print("[!] best_model.pth 없음 - /train 엔드포인트로 학습 먼저 필요")


# ─────────────────────────────────────────────
# 이미지 전처리
# ─────────────────────────────────────────────
def preprocess_image(image_data: str) -> torch.Tensor:
    """
    Base64 PNG → 28×28 그레이스케일 → MNIST 정규화 텐서
    """
    # base64 디코딩
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    img_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGBA')

    # 흰 배경에 합성 (캔버스는 투명 배경일 수 있음)
    background = Image.new('RGB', img.size, (0, 0, 0))
    background.paste(img, mask=img.split()[3])  # alpha 채널로 마스크
    img = background.convert('L')  # 그레이스케일

    # 28×28로 리사이즈 (LANCZOS 필터로 품질 유지)
    img = img.resize((28, 28), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32)

    # MNIST는 검정 배경 + 흰 글씨 → 캔버스가 반대면 반전
    # 평균 픽셀값이 128 이상이면 반전 (밝은 배경에 어두운 글씨)
    if arr.mean() > 128:
        arr = 255.0 - arr

    arr = arr / 255.0
    arr = (arr - 0.1307) / 0.3081  # MNIST 정규화

    tensor = torch.FloatTensor(arr).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
    return tensor.to(DEVICE)


# ─────────────────────────────────────────────
# 빠른 학습 (모델 미존재 시)
# ─────────────────────────────────────────────
MNIST_MIRROR = {
    'train_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
    'test_images':  'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels':  'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
}

def quick_train(epochs=30, batch_size=128, callback=None):
    """서버 내장 강화 학습 (ResNet + 데이터 증강)"""
    from torch.utils.data import DataLoader, Dataset
    import torch.optim as optim

    data_dir = os.path.join(os.path.dirname(__file__), 'mnist_data')
    os.makedirs(data_dir, exist_ok=True)

    def download():
        for key, url in MNIST_MIRROR.items():
            gz = os.path.join(data_dir, key + '.gz')
            raw = os.path.join(data_dir, key)
            if os.path.exists(raw): continue
            if callback: callback(f"다운로드: {key}")
            urllib.request.urlretrieve(url, gz)
            with gzip.open(gz,'rb') as fi, open(raw,'wb') as fo:
                shutil.copyfileobj(fi, fo)
            os.remove(gz)

    def _load_images(path):
        with open(path,'rb') as f:
            _,n,r,c = struct.unpack('>IIII', f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n,r,c)

    def _load_labels(path):
        with open(path,'rb') as f:
            _,n = struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    download()

    class DS(Dataset):
        MEAN, STD = 0.1307, 0.3081
        def __init__(self, imgs, lbls, aug=False):
            self.raw = imgs.astype(np.float32)/255.0
            self.imgs = (torch.FloatTensor(self.raw).unsqueeze(1)-self.MEAN)/self.STD
            self.lbls = torch.LongTensor(lbls.astype(np.int64))
            self.aug  = aug
        def __len__(self): return len(self.lbls)
        def __getitem__(self, i):
            img = self.imgs[i].clone()
            if self.aug:
                # 랜덤 회전
                if np.random.random() < 0.6:
                    angle = np.random.uniform(-15,15)*math.pi/180
                    c,s = math.cos(angle),math.sin(angle)
                    th = torch.tensor([[c,-s,0],[s,c,0]],dtype=torch.float32).unsqueeze(0)
                    grid = F.affine_grid(th,img.unsqueeze(0).shape,align_corners=False)
                    img = F.grid_sample(img.unsqueeze(0),grid,align_corners=False,
                                        padding_mode='zeros').squeeze(0)
                # 시프트
                if np.random.random() < 0.5:
                    img = torch.roll(img,(np.random.randint(-3,4),np.random.randint(-3,4)),(1,2))
                # Cutout
                if np.random.random() < 0.3:
                    cy,cx = np.random.randint(28),np.random.randint(28)
                    img[...,max(0,cy-5):min(28,cy+5),max(0,cx-5):min(28,cx+5)] = 0
            return img, self.lbls[i]

    tr_imgs = _load_images(os.path.join(data_dir,'train_images'))
    tr_lbls = _load_labels(os.path.join(data_dir,'train_labels'))
    te_imgs = _load_images(os.path.join(data_dir,'test_images'))
    te_lbls = _load_labels(os.path.join(data_dir,'test_labels'))

    train_loader = DataLoader(DS(tr_imgs,tr_lbls,aug=True),  batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(DS(te_imgs,te_lbls,aug=False), batch_size=batch_size)

    global model, TRAINED
    model = MNISTNet(drop=0.3).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3,
        steps_per_epoch=len(train_loader), epochs=epochs,
        pct_start=0.1, anneal_strategy='cos',
        div_factor=25, final_div_factor=1e4)
    criterion = nn.CrossEntropyLoss()

    use_amp = DEVICE.type == 'cuda'
    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler() if use_amp else None

    for ep in range(1, epochs+1):
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            if scaler:
                with autocast():
                    loss = criterion(model(imgs), lbls)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            else:
                criterion(model(imgs), lbls).backward()
                optimizer.step()
            scheduler.step()

        model.eval(); correct = total = 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                correct += model(imgs).argmax(1).eq(lbls).sum().item()
                total   += lbls.size(0)
        acc = 100.*correct/total
        msg = f"Epoch {ep}/{epochs}  정확도: {acc:.2f}%"
        if callback: callback(msg)
        print(msg)

    torch.save({'model_state_dict': model.state_dict(), 'val_acc': acc}, MODEL_PATH)
    model.eval(); TRAINED = True
    return acc


# ─────────────────────────────────────────────
# API 엔드포인트
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    if not TRAINED:
        return jsonify({'error': '모델이 학습되지 않았습니다. /api/train 을 먼저 호출하세요.'}), 400

    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': '이미지 데이터가 없습니다.'}), 400

    try:
        tensor = preprocess_image(data['image'])
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred = int(probs.argmax())
        confidence = float(probs[pred])
        all_probs = {str(i): round(float(p)*100, 2) for i, p in enumerate(probs)}

        return jsonify({
            'prediction': pred,
            'confidence': round(confidence*100, 2),
            'probabilities': all_probs
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'trained': TRAINED,
        'device': str(DEVICE),
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH)
    })


@app.route('/api/train', methods=['POST'])
def train_endpoint():
    """모델 미존재 시 서버에서 직접 학습"""
    if TRAINED:
        return jsonify({'message': '이미 학습된 모델이 있습니다.', 'trained': True})

    data = request.json or {}
    epochs = data.get('epochs', 30)

    import threading
    log = []
    done = {'acc': None, 'error': None}

    def run():
        try:
            acc = quick_train(epochs=epochs, callback=lambda m: log.append(m))
            done['acc'] = acc
        except Exception as e:
            done['error'] = str(e)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=600)

    if done['error']:
        return jsonify({'error': done['error']}), 500
    return jsonify({'message': '학습 완료', 'accuracy': done['acc'], 'log': log})


if __name__ == '__main__':
    print("=" * 50)
    print("  MNIST 손글씨 인식 웹 서버")
    print(f"  디바이스: {DEVICE}")
    print("  http://localhost:5000 에서 실행")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)