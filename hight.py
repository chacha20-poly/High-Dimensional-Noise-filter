import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

import torch

def add_salt_and_pepper_noise(img, scale=0.1):
    """
    img: [B, 1, H, W] 画像テンソル（0~1）
    amount: 画像全体に占めるノイズの割合（0.0~1.0）
    """
    noisy = img.clone()
    num_pixels = int(scale * img.numel())
    
    # 全ピクセルインデックスをランダム選択
    idx = torch.randint(0, img.numel(), (num_pixels,))
    
    # 半分をソルト(1)、半分をペッパー(0)
    half = num_pixels // 2
    noisy.view(-1)[idx[:half]] = 1.0  # ソルト
    noisy.view(-1)[idx[half:]] = 0.0  # ペッパー
    
    return noisy


# -------------------------
# 実数MLP（通常学習）
# -------------------------
class RealMLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# -------------------------
# 複素線形層
# -------------------------
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.real = nn.Linear(in_features, out_features)
        self.imag = nn.Linear(in_features, out_features)
    def forward(self, x):
        xr, xi = x.real, x.imag
        real = self.real(xr) - self.imag(xi)
        imag = self.real(xi) + self.imag(xr)
        return torch.complex(real, imag)

# -------------------------
# 複素ReLU
# -------------------------
class ComplexReLU(nn.Module):
    def forward(self, x):
        return torch.complex(torch.relu(x.real), torch.relu(x.imag))

# -------------------------
# 複素MLP（虚部に x^2 を入れ高次元化）
# -------------------------
class ComplexMLP(nn.Module):
    def __init__(self, n_dim=28*28, hidden_dim=2048):
        super().__init__()
        self.fc1 = ComplexLinear(n_dim, hidden_dim)
        self.fc2 = ComplexLinear(hidden_dim, hidden_dim)
        self.fc3 = ComplexLinear(hidden_dim, n_dim)
        self.act = ComplexReLU()

    def forward(self, x_real):
        # 入力が実数なら高次元化して複素数に
        if torch.is_floating_point(x_real):
            # 高次元化: x^2, x^3, sin(x) を組み合わせる
            x_imag = x_real**2 + x_real**3 + torch.sin(x_real * np.pi)
            x = torch.complex(x_real, x_imag)
        else:
            x = x_real  # すでに複素数ならそのまま

        # 複素MLP
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x.real)


# -------------------------
# データ準備
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False)

# -------------------------
# 学習関数
# -------------------------
def train_model(model, loader, epochs=5, lr=1e-3, noise_scale=0.5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in loader:
            x = x.to(device).view(x.size(0), -1)
            noisy = add_salt_and_pepper_noise(x, scale=noise_scale).to(device)
            optimizer.zero_grad()
            out = model(noisy)
            loss = criterion(out, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: train_loss={total_loss/len(loader):.6f}")

# -------------------------
# 評価関数
# -------------------------
def evaluate_model(model, loader, noise_scale=0.5):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device).view(x.size(0), -1)
            noisy = add_salt_and_pepper_noise(x, scale=noise_scale).to(device)
            out = model(noisy)
            total_loss += criterion(out, x).item() * x.size(0)
    mse = total_loss / len(loader.dataset)
    psnr = 10 * np.log10(1.0 / mse)
    return mse, psnr

# -------------------------
# モデル作成
# -------------------------
real_model = RealMLP().to(device)
complex_model = ComplexMLP().to(device)

# -------------------------
# 学習
# -------------------------
print("=== Training Real MLP ===")
train_model(real_model, train_loader, epochs=5, noise_scale=1.0)

print("\n=== Training Complex MLP ===")
train_model(complex_model, train_loader, epochs=5, noise_scale=1.0)

# -------------------------
# 評価
# -------------------------


# -------------------------
# 画像比較
# -------------------------
sample_img, _ = test_dataset[0]
sample_img = sample_img.unsqueeze(0).to(device)
noisy_sample = add_salt_and_pepper_noise(sample_img, scale=0.5).to(device)

# 比較したいノイズレベルリスト
# 画像番号リスト
import matplotlib.pyplot as plt
import torch

# 表示する画像番号
# 画像番号リスト
# ノイズレベルリスト
image_indices = list(range(1, 10))  # 表示する画像番号
noise_levels = [0.1, 0.2, 0.5, 0.7, 0.8]

for idx in image_indices:
    sample_img, _ = test_dataset[idx]
    sample_img = sample_img.unsqueeze(0).to(device)

    n_levels = len(noise_levels)
    plt.figure(figsize=((n_levels+1)*3, 3))  # 左端オリジナル分+ノイズ列

    # -------------------------
    # 左端にオリジナル表示
    # -------------------------
    
    plt.subplot(1, (n_levels+1)*3, 1)
    plt.imshow(sample_img.cpu().squeeze(), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    for i, noise_std in enumerate(noise_levels):
        noisy = add_salt_and_pepper_noise(sample_img, scale=noise_std).to(device)  # 例: ソルト＆ペッパー
        mse_real, psnr_real = evaluate_model(real_model, test_loader, noise_scale=noise_std)
        mse_complex, psnr_complex = evaluate_model(complex_model, test_loader, noise_scale=noise_std)
        print(noise_std)
        print(f"\nReal MLP: MSE={mse_real:.5f}, PSNR={psnr_real:.2f} dB")
        print(f"Complex MLP: MSE={mse_complex:.5f}, PSNR={psnr_complex:.2f} dB")
        with torch.no_grad():
            real_out = real_model(noisy.view(1, -1)).view_as(sample_img)
            complex_out = complex_model(noisy.view(1, -1)).view_as(sample_img)

        # CPUに戻す
        noisy_cpu = noisy.cpu()
        real_cpu = real_out.cpu()
        complex_cpu = complex_out.cpu()

        # -------------------------
        # プロット列番号（オリジナルの右側から）
        # -------------------------
        base_col = 1 + i*3
        plt.subplot(1, (n_levels+1)*3, base_col+1)
        plt.imshow(noisy_cpu.squeeze(), cmap='gray')
        plt.title(f"Noise {noise_std}")
        plt.axis('off')

        plt.subplot(1, (n_levels+1)*3, base_col+2)
        plt.imshow(real_cpu.squeeze(), cmap='gray')
        plt.title("Real MLP")
        plt.axis('off')

        plt.subplot(1, (n_levels+1)*3, base_col+3)
        plt.imshow(complex_cpu.squeeze(), cmap='gray')
        plt.title("Complex MLP")
        plt.axis('off')

    plt.tight_layout()
    plt.show(block=True)  # ウィンドウを閉じるまで待機