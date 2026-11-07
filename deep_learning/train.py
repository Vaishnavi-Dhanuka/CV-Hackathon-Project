
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class LOLDataset(Dataset):
    def __init__(self, root, split="Train", patch_size=256):
        self.low_dir = Path(root) / split / "Low"
        self.normal_dir = Path(root) / split / "Normal"
        self.patch_size = patch_size

        self.low_files = sorted(self.low_dir.glob("*.png"))
        self.normal_files = sorted(self.normal_dir.glob("*.png"))

        if not self.low_files or len(self.low_files) != len(self.normal_files):
            raise RuntimeError(
                f"Invalid {split} dataset: "
                f"{len(self.low_files)} low, {len(self.normal_files)} normal images"
            )

    def __len__(self):
        return len(self.low_files)

    def __getitem__(self, index):
        low = Image.open(self.low_files[index]).convert("RGB")
        normal = Image.open(self.normal_files[index]).convert("RGB")

        low = np.asarray(low, dtype=np.float32) / 255.0
        normal = np.asarray(normal, dtype=np.float32) / 255.0

        h, w = low.shape[:2]
        p = self.patch_size

        if h >= p and w >= p:
            top = random.randint(0, h - p)
            left = random.randint(0, w - p)
            low = low[top:top+p, left:left+p]
            normal = normal[top:top+p, left:left+p]

        if random.random() < 0.5:
            low = np.fliplr(low).copy()
            normal = np.fliplr(normal).copy()

        if random.random() < 0.5:
            low = np.flipud(low).copy()
            normal = np.flipud(normal).copy()

        low = torch.from_numpy(low.copy()).permute(2, 0, 1)
        normal = torch.from_numpy(normal.copy()).permute(2, 0, 1)

        return low, normal


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LowLightNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = ConvBlock(3, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        self.output = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.output(d1))


def evaluate(model, loader, device):
    model.eval()
    total_l1 = 0.0
    total_psnr = 0.0
    count = 0

    with torch.no_grad():
        for low, target in loader:
            low = low.to(device)
            target = target.to(device)

            output = model(low)
            mse = F.mse_loss(output, target).item()
            psnr = 10 * np.log10(1.0 / max(mse, 1e-10))

            total_l1 += F.l1_loss(output, target).item()
            total_psnr += psnr
            count += 1

    return total_l1 / count, total_psnr / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/Kaggle/archive/LOL-v2/Real_captured")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_set = LOLDataset(args.data, "Train", args.patch_size)
    test_set = LOLDataset(args.data, "Test", args.patch_size)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = LowLightNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    checkpoint_dir = Path("results/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_psnr = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for low, target in progress:
            low = low.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(low)
                loss = F.l1_loss(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        val_l1, val_psnr = evaluate(model, test_loader, device)
        train_loss = running_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}: "
            f"Train L1={train_loss:.4f} | "
            f"Test L1={val_l1:.4f} | "
            f"Test PSNR={val_psnr:.2f} dB"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "psnr": val_psnr,
            },
            checkpoint_dir / "latest.pth",
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(
                model.state_dict(),
                checkpoint_dir / "best_model.pth",
            )
            print(f"Saved best model: {best_psnr:.2f} dB")

    print("Training complete. Best PSNR:", best_psnr)


if __name__ == "__main__":
    main()
