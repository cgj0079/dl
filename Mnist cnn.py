"""
MNIST 손글씨 인식 - 고성능 CNN (GPU 최적화 버전)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[강화된 항목]
 • 더 깊은 ResNet 스타일 모델 (Residual Block + Squeeze-Excitation)
 • 강력한 데이터 증강 (ElasticDistortion, Cutout, RandomRotate, RandomScale)
 • Mixup 학습 기법
 • Label Smoothing Loss
 • OneCycleLR (워밍업 포함 코사인 어닐링)
 • Mixed Precision Training (AMP) — GPU 속도 1.5~2x 향상
 • TTA (Test Time Augmentation) — 추론 앙상블
 • Early Stopping + 체크포인트 재개
 • 목표: GPU 환경 99.5%+ 달성
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import struct, os, urllib.request, gzip, shutil, time, argparse, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════
# 1. MNIST 데이터 다운로드 & 파싱
# ══════════════════════════════════════════════════════════
MIRROR_URLS = {
    'train_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
    'test_images':  'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels':  'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
}
YANN_URLS = {k: v.replace('ossci-datasets.s3.amazonaws.com/mnist',
                            'yann.lecun.com/exdb/mnist')
             for k, v in MIRROR_URLS.items()}

def download_mnist(data_dir='./mnist_data'):
    os.makedirs(data_dir, exist_ok=True)
    for key in MIRROR_URLS:
        raw = os.path.join(data_dir, key)
        if os.path.exists(raw):
            print(f'  [OK] {key}'); continue
        gz = raw + '.gz'
        print(f'  [DL] {key} 다운로드 중...')
        for url in [MIRROR_URLS[key], YANN_URLS[key]]:
            try:
                urllib.request.urlretrieve(url, gz); break
            except Exception as e:
                print(f'       실패({e}), 미러 시도...')
        with gzip.open(gz,'rb') as fi, open(raw,'wb') as fo:
            shutil.copyfileobj(fi, fo)
        os.remove(gz)
        print(f'       완료!')

def load_images(path):
    with open(path,'rb') as f:
        _, n, r, c = struct.unpack('>IIII', f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c)

def load_labels(path):
    with open(path,'rb') as f:
        _, n = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


# ══════════════════════════════════════════════════════════
# 2. 강력한 데이터 증강
# ══════════════════════════════════════════════════════════
class ElasticDistortion:
    """탄성 변형: 손글씨 스타일 다양화에 매우 효과적"""
    def __init__(self, alpha=34, sigma=4):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img: np.ndarray) -> np.ndarray:
        H, W = img.shape
        dx = np.random.uniform(-1, 1, (H, W)).astype(np.float32)
        dy = np.random.uniform(-1, 1, (H, W)).astype(np.float32)
        dx = self._gauss(dx, self.sigma) * self.alpha
        dy = self._gauss(dy, self.sigma) * self.alpha
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        map_x = np.clip(x + dx, 0, W-1).astype(np.float32)
        map_y = np.clip(y + dy, 0, H-1).astype(np.float32)
        return self._remap(img, map_x, map_y)

    def _gauss(self, img, sigma):
        k = int(6*sigma+1)|1
        ax = np.arange(-k//2+1, k//2+1, dtype=np.float32)
        kernel = np.exp(-0.5*(ax/sigma)**2); kernel /= kernel.sum()
        out = np.apply_along_axis(lambda r: np.convolve(r, kernel, 'same'), 0, img)
        return np.apply_along_axis(lambda r: np.convolve(r, kernel, 'same'), 1, out)

    def _remap(self, img, mx, my):
        H, W = img.shape
        x0 = np.floor(mx).astype(int); x1 = np.clip(x0+1,0,W-1)
        y0 = np.floor(my).astype(int); y1 = np.clip(y0+1,0,H-1)
        wx = mx-x0; wy = my-y0
        return (img[y0,x0]*(1-wy)*(1-wx) + img[y0,x1]*(1-wy)*wx +
                img[y1,x0]*wy*(1-wx)     + img[y1,x1]*wy*wx).astype(np.float32)


def rotate_fast(img: torch.Tensor, max_angle=15) -> torch.Tensor:
    angle = np.random.uniform(-max_angle, max_angle)
    rad = torch.tensor(angle * math.pi / 180)
    c, s = torch.cos(rad).item(), torch.sin(rad).item()
    theta = torch.tensor([[c,-s,0],[s,c,0]], dtype=torch.float32).unsqueeze(0)
    grid = F.affine_grid(theta, img.unsqueeze(0).shape, align_corners=False)
    return F.grid_sample(img.unsqueeze(0), grid,
                         align_corners=False, padding_mode='zeros').squeeze(0)

def scale_shift(img: torch.Tensor) -> torch.Tensor:
    sc = np.random.uniform(0.85, 1.15)
    tx = np.random.randint(-3, 4) / 14.0
    ty = np.random.randint(-3, 4) / 14.0
    s = 1/sc
    theta = torch.tensor([[s,0,tx],[0,s,ty]], dtype=torch.float32).unsqueeze(0)
    grid = F.affine_grid(theta, img.unsqueeze(0).shape, align_corners=False)
    return F.grid_sample(img.unsqueeze(0), grid,
                         align_corners=False, padding_mode='zeros').squeeze(0)

def cutout(img: torch.Tensor, length=10) -> torch.Tensor:
    H, W = img.shape[-2], img.shape[-1]
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = max(0,cy-length//2), min(H,cy+length//2)
    x1, x2 = max(0,cx-length//2), min(W,cx+length//2)
    img = img.clone()
    img[..., y1:y2, x1:x2] = 0
    return img


class MNISTDataset(Dataset):
    MEAN, STD = 0.1307, 0.3081

    def __init__(self, images, labels, augment=False, elastic_prob=0.5):
        raw = images.astype(np.float32) / 255.0
        self.raw_np = raw
        self.images = (torch.FloatTensor(raw).unsqueeze(1) - self.MEAN) / self.STD
        self.labels = torch.LongTensor(labels.astype(np.int64))
        self.augment = augment
        self.elastic = ElasticDistortion(alpha=34, sigma=4)
        self.ep = elastic_prob

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].clone()
        lbl = self.labels[idx]
        if not self.augment:
            return img, lbl

        r = np.random.random

        # 1) 탄성 변형 (ep 확률)
        if r() < self.ep:
            d = self.elastic(self.raw_np[idx])
            img = (torch.FloatTensor(d).unsqueeze(0) - self.MEAN) / self.STD

        # 2) 랜덤 회전 ±15° (70%)
        if r() < 0.70: img = rotate_fast(img, 15)

        # 3) 스케일 + 시프트 (60%)
        if r() < 0.60: img = scale_shift(img)

        # 4) Cutout (40%)
        if r() < 0.40: img = cutout(img, 10)

        # 5) 가우시안 노이즈 (30%)
        if r() < 0.30: img = img + torch.randn_like(img) * 0.1

        return img, lbl


# ══════════════════════════════════════════════════════════
# 3. ResNet + Squeeze-Excitation 모델
# ══════════════════════════════════════════════════════════
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
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch))
        self.se  = SEBlock(ch)
        self.dp  = nn.Dropout2d(drop)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.se(self.dp(self.net(x))) + x)


class MNISTNet(nn.Module):
    """
    Stem → Stage1(64ch, 2×Res) → Stage2(128ch, 3×Res) → Stage3(256ch, 2×Res)
    → GlobalAvgPool → FC(256→256) → FC(256→10)
    파라미터: ~820K
    """
    def __init__(self, drop=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1,bias=False),
            nn.BatchNorm2d(32), nn.GELU())

        def stage(ci, co, n, dp):
            layers = [nn.Conv2d(ci,co,3,padding=1,bias=False),
                      nn.BatchNorm2d(co), nn.GELU()]
            layers += [ResBlock(co, dp) for _ in range(n)]
            return nn.Sequential(*layers)

        self.s1 = nn.Sequential(stage(32, 64,  2, 0.10), nn.MaxPool2d(2))  # 28→14
        self.s2 = nn.Sequential(stage(64, 128, 3, 0.15), nn.MaxPool2d(2))  # 14→7
        self.s3 = nn.Sequential(stage(128,256, 2, 0.20), nn.AdaptiveAvgPool2d(1))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(256,10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight,1); nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.s3(self.s2(self.s1(self.stem(x)))))


# ══════════════════════════════════════════════════════════
# 4. Label Smoothing + Mixup
# ══════════════════════════════════════════════════════════
class LabelSmoothLoss(nn.Module):
    def __init__(self, classes=10, smooth=0.1):
        super().__init__()
        self.smooth = smooth; self.cls = classes

    def forward(self, pred, tgt):
        one_hot = torch.full_like(pred, self.smooth/(self.cls-1))
        one_hot.scatter_(1, tgt.unsqueeze(1), 1-self.smooth)
        return -(one_hot * F.log_softmax(pred,1)).sum(1).mean()


def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x+(1-lam)*x[idx], y, y[idx], lam

def mixup_loss(crit, out, ya, yb, lam):
    return lam*crit(out,ya)+(1-lam)*crit(out,yb)


# ══════════════════════════════════════════════════════════
# 5. TTA 추론
# ══════════════════════════════════════════════════════════
@torch.no_grad()
def tta_predict(model, imgs, device):
    imgs = imgs.to(device); model.eval()
    preds = [F.softmax(model(imgs), 1)]

    for angle in [-10, -5, 5, 10]:
        rad = torch.tensor(angle*math.pi/180)
        c, s = torch.cos(rad).item(), torch.sin(rad).item()
        th = torch.tensor([[c,-s,0],[s,c,0]],dtype=torch.float32
                          ).unsqueeze(0).expand(imgs.size(0),-1,-1)
        grid = F.affine_grid(th, imgs.shape, align_corners=False)
        aug = F.grid_sample(imgs, grid, align_corners=False, padding_mode='zeros')
        preds.append(F.softmax(model(aug), 1))

    for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1),(0,-2),(0,2),(-2,0),(2,0)]:
        preds.append(F.softmax(model(torch.roll(imgs,(dy,dx),(2,3))), 1))

    return torch.stack(preds).mean(0)


# ══════════════════════════════════════════════════════════
# 6. 학습 & 평가
# ══════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, crit, device,
                scheduler, scaler, alpha):
    model.train()
    tloss = tcorr = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        use_mx = (alpha > 0) and (np.random.random() < 0.5)
        if use_mx:
            imgs_m, ya, yb, lam = mixup(imgs, labels, alpha)
        else:
            imgs_m, ya, yb, lam = imgs, labels, labels, 1.0

        optimizer.zero_grad()
        if scaler:
            with autocast():
                out  = model(imgs_m)
                loss = mixup_loss(crit, out, ya, yb, lam)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            out  = model(imgs_m)
            loss = mixup_loss(crit, out, ya, yb, lam)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        tloss += loss.item() * imgs.size(0)

        with torch.no_grad():
            _, pred = model(imgs).max(1)
            tcorr += pred.eq(labels).sum().item()
        total += imgs.size(0)

    return tloss/total, 100.*tcorr/total


def evaluate(model, loader, crit, device, tta=False):
    model.eval()
    tloss = tcorr = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if tta:
                probs = tta_predict(model, imgs, device)
                _, pred = probs.max(1)
                loss = crit(model(imgs), labels)
            else:
                out = model(imgs)
                loss = crit(out, labels)
                _, pred = out.max(1)
            tloss += loss.item()*imgs.size(0)
            tcorr += pred.eq(labels).sum().item()
            total += imgs.size(0)
    return tloss/total, 100.*tcorr/total


# ══════════════════════════════════════════════════════════
# 7. 시각화
# ══════════════════════════════════════════════════════════
def _dark_ax(ax):
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='#aaa')
    for sp in ax.spines.values(): sp.set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_history(tl, vl, ta, va, lrs, path):
    fig, axes = plt.subplots(1,3,figsize=(16,4),facecolor='#111')
    for ax in axes: _dark_ax(ax)
    eps = range(1, len(tl)+1)

    axes[0].plot(eps,tl,color='#00f5a0',label='Train',lw=2)
    axes[0].plot(eps,vl,color='#00d4ff',label='Val',lw=2,ls='--')
    axes[0].set_title('Loss',color='#eee'); axes[0].legend(facecolor='#222',labelcolor='#ddd')
    axes[0].grid(True,color='#222')

    best = max(va); bi = va.index(best)+1
    axes[1].plot(eps,ta,color='#00f5a0',label='Train',lw=2)
    axes[1].plot(eps,va,color='#00d4ff',label='Val',lw=2,ls='--')
    axes[1].axhline(best,color='#ff3e6c',ls=':',lw=1.5,label=f'Best {best:.2f}%')
    axes[1].scatter([bi],[best],color='#ff3e6c',s=60,zorder=5)
    axes[1].set_title('Accuracy (%)',color='#eee'); axes[1].legend(facecolor='#222',labelcolor='#ddd')
    axes[1].grid(True,color='#222')

    if lrs:
        xs = np.linspace(0, len(lrs), min(len(lrs),2000), dtype=int)
        axes[2].semilogy([lrs[i] for i in xs],color='#ffaa00',lw=1.5)
        axes[2].set_title('Learning Rate (log)',color='#eee')
        axes[2].grid(True,color='#222')

    plt.suptitle('MNIST 학습 기록',color='#fff',fontsize=14)
    plt.tight_layout()
    plt.savefig(path,dpi=130,bbox_inches='tight',facecolor='#111')
    plt.close(); print(f'[OK] {path}')


def visualize_predictions(model, ds, device, n=25, path='predictions.png'):
    model.eval()
    idx = np.random.choice(len(ds), n, replace=False)
    fig, axes = plt.subplots(5,5,figsize=(10,10),facecolor='#111')
    with torch.no_grad():
        for i, di in enumerate(idx):
            img, lbl = ds[di]
            probs = tta_predict(model, img.unsqueeze(0), device)
            pred = probs.argmax(1).item(); conf = probs.max().item()
            ax = axes[i//5][i%5]
            ax.imshow(img.squeeze().numpy()*0.3081+0.1307, cmap='gray', vmin=0, vmax=1)
            c = '#00f5a0' if pred==lbl.item() else '#ff3e6c'
            ax.set_title(f'예:{pred} 실:{lbl.item()}\n{conf:.1%}', color=c, fontsize=8)
            ax.axis('off'); ax.set_facecolor('#000')
    plt.suptitle('예측 결과 (TTA)',color='#fff',fontsize=12)
    plt.tight_layout()
    plt.savefig(path,dpi=130,bbox_inches='tight',facecolor='#111')
    plt.close(); print(f'[OK] {path}')


def show_cm(model, loader, device, path):
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            all_p.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
            all_t.extend(lbls.numpy())
    cm = np.zeros((10,10),dtype=int)
    for t,p in zip(all_t,all_p): cm[t][p]+=1
    fig, ax = plt.subplots(figsize=(9,8),facecolor='#111')
    _dark_ax(ax)
    im = ax.imshow(cm,cmap='YlOrRd')
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel('예측값',color='#ccc'); ax.set_ylabel('실제값',color='#ccc')
    ax.set_title('혼동 행렬',color='#fff',fontsize=13)
    for i in range(10):
        for j in range(10):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=9,fontweight='bold',
                    color='white' if cm[i,j]>cm.max()*0.5 else 'black')
    plt.colorbar(im,ax=ax)
    print('\n  클래스별 정확도:')
    for i in range(10):
        print(f'    {i}: {cm[i,i]/cm[i].sum()*100:.2f}%')
    plt.tight_layout()
    plt.savefig(path,dpi=130,bbox_inches='tight',facecolor='#111')
    plt.close(); print(f'[OK] {path}')


# ══════════════════════════════════════════════════════════
# 8. 메인
# ══════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description='MNIST 고성능 CNN',
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--epochs',       type=int,   default=60)
    p.add_argument('--batch-size',   type=int,   default=128,
                   help='GPU 8GB+: 256 권장')
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--drop',         type=float, default=0.3)
    p.add_argument('--mixup-alpha',  type=float, default=0.4)
    p.add_argument('--label-smooth', type=float, default=0.1)
    p.add_argument('--elastic-prob', type=float, default=0.5)
    p.add_argument('--patience',     type=int,   default=15,
                   help='Early Stop 인내 에포크')
    p.add_argument('--no-amp',       action='store_true')
    p.add_argument('--no-tta',       action='store_true')
    p.add_argument('--data-dir',     type=str,   default='./mnist_data')
    p.add_argument('--save-dir',     type=str,   default='./output')
    p.add_argument('--resume',       type=str,   default='')
    p.add_argument('--device',       type=str,   default='auto')
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.device == 'auto':
        device = (torch.device('cuda')  if torch.cuda.is_available()  else
                  torch.device('mps')   if torch.backends.mps.is_available() else
                  torch.device('cpu'))
    else:
        device = torch.device(args.device)

    use_amp = (not args.no_amp) and (device.type == 'cuda')

    print('╔' + '═'*55 + '╗')
    print('║     MNIST 고성능 CNN  (ResNet + SE + AMP + TTA)     ║')
    print('╠' + '═'*55 + '╣')
    print(f'║  디바이스    : {str(device):<40}║')
    if device.type == 'cuda':
        gn = torch.cuda.get_device_name(0)[:38]
        gm = torch.cuda.get_device_properties(0).total_memory/1e9
        print(f'║  GPU         : {gn:<40}║')
        print(f'║  GPU 메모리  : {gm:<40.1f}║')
    print(f'║  AMP 가속    : {str(use_amp):<40}║')
    print(f'║  에포크      : {args.epochs:<40}║')
    print(f'║  배치 크기   : {args.batch_size:<40}║')
    print(f'║  학습률(max) : {args.lr:<40}║')
    print(f'║  Mixup alpha : {args.mixup_alpha:<40}║')
    print(f'║  LabelSmooth : {args.label_smooth:<40}║')
    print(f'║  탄성변형확률: {args.elastic_prob:<40}║')
    print(f'║  Early Stop  : {args.patience} 에포크{"":<34}║')
    print(f'║  TTA 추론    : {str(not args.no_tta):<40}║')
    print('╚' + '═'*55 + '╝\n')

    # 데이터
    print('데이터 준비...')
    download_mnist(args.data_dir)
    tr_img = load_images(os.path.join(args.data_dir,'train_images'))
    tr_lbl = load_labels(os.path.join(args.data_dir,'train_labels'))
    te_img = load_images(os.path.join(args.data_dir,'test_images'))
    te_lbl = load_labels(os.path.join(args.data_dir,'test_labels'))
    print(f'  학습: {len(tr_img):,}개  테스트: {len(te_img):,}개\n')

    train_ds = MNISTDataset(tr_img, tr_lbl, augment=True, elastic_prob=args.elastic_prob)
    test_ds  = MNISTDataset(te_img, te_lbl, augment=False)

    nw = 4 if device.type=='cuda' else 0
    pin = device.type=='cuda'
    train_ld = DataLoader(train_ds, args.batch_size, shuffle=True,
                          num_workers=nw, pin_memory=pin,
                          persistent_workers=(nw>0))
    test_ld  = DataLoader(test_ds, args.batch_size, shuffle=False,
                          num_workers=nw, pin_memory=pin,
                          persistent_workers=(nw>0))

    # 모델
    model = MNISTNet(drop=args.drop).to(device)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'모델 파라미터: {n:,} ({n/1e6:.2f}M)\n')

    crit = (LabelSmoothLoss(smooth=args.label_smooth)
            if args.label_smooth>0 else nn.CrossEntropyLoss())
    opt  = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    spe = len(train_ld)
    sched = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr,
        steps_per_epoch=spe, epochs=args.epochs,
        pct_start=0.1, anneal_strategy='cos',
        div_factor=25, final_div_factor=1e4)
    scaler = GradScaler() if use_amp else None

    # 재개
    start_ep = 1; best_acc = 0.0
    best_path = os.path.join(args.save_dir, 'best_model.pth')
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        opt.load_state_dict(ck['optimizer_state_dict'])
        start_ep = ck.get('epoch',0)+1
        best_acc  = ck.get('val_acc',0.0)
        print(f'[→] 재개: epoch {start_ep}, best={best_acc:.2f}%\n')

    # 학습
    tl_h, vl_h, ta_h, va_h, lr_h = [], [], [], [], []
    no_imp = 0
    t0_all = time.time()

    print(f'{"─"*72}')
    print(f' {"Epoch":>5}  {"TrainAcc":>8}  {"TrainLoss":>9}  '
          f'{"ValAcc":>8}  {"ValLoss":>8}  {"LR":>8}  {"Time":>5}')
    print(f'{"─"*72}')

    for ep in range(start_ep, args.epochs+1):
        t0 = time.time()

        tr_loss, tr_acc = train_epoch(
            model, train_ld, opt, crit, device, sched, scaler, args.mixup_alpha)

        do_tta = (not args.no_tta) and (ep==args.epochs or ep%10==0)
        va_loss, va_acc = evaluate(model, test_ld, crit, device, tta=do_tta)

        cur_lr = opt.param_groups[0]['lr']
        lr_h.append(cur_lr)
        tl_h.append(tr_loss); vl_h.append(va_loss)
        ta_h.append(tr_acc);  va_h.append(va_acc)

        is_best = va_acc > best_acc
        if is_best:
            best_acc = va_acc; no_imp = 0
            torch.save({'epoch':ep,'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':opt.state_dict(),
                        'val_acc':va_acc,'val_loss':va_loss}, best_path)
        else:
            no_imp += 1

        tta_tag = '[TTA]' if do_tta else '     '
        star    = ' ★' if is_best else ''
        print(f' {ep:5d}  {tr_acc:7.3f}%  {tr_loss:9.5f}  '
              f'{va_acc:7.3f}%{tta_tag}  {va_loss:8.5f}  '
              f'{cur_lr:.2e}  {time.time()-t0:4.1f}s{star}')

        if no_imp >= args.patience:
            print(f'\n[Early Stop] {args.patience} 에포크 개선 없음')
            break

        if ep%10==0:
            plot_history(tl_h,vl_h,ta_h,va_h,lr_h,
                         os.path.join(args.save_dir,f'history_ep{ep}.png'))

    print(f'{"─"*72}')
    total_t = (time.time()-t0_all)/60
    print(f'\n학습 완료! 총 소요: {total_t:.1f}분')
    print(f'최고 검증 정확도: {best_acc:.4f}%')

    # 최종 평가
    ck = torch.load(best_path, map_location=device)
    model.load_state_dict(ck['model_state_dict'])

    print('\n최종 평가 (TTA 포함)...')
    _, plain = evaluate(model, test_ld, crit, device, tta=False)
    _, tta   = evaluate(model, test_ld, crit, device, tta=True)
    print(f'  일반 정확도 : {plain:.4f}%')
    print(f'  TTA  정확도 : {tta:.4f}%  (오류율: {100-tta:.4f}%)')

    # 시각화
    plot_history(tl_h,vl_h,ta_h,va_h,lr_h,
                 os.path.join(args.save_dir,'training_history.png'))
    visualize_predictions(model, test_ds, device,
                          path=os.path.join(args.save_dir,'predictions.png'))
    show_cm(model, test_ld, device,
            path=os.path.join(args.save_dir,'confusion_matrix.png'))

    print(f'\n저장 위치: {os.path.abspath(args.save_dir)}/')
    print('  best_model.pth / training_history.png / predictions.png / confusion_matrix.png')


if __name__ == '__main__':
    main()