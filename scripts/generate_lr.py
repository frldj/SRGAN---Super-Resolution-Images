"""
Generate LR images from HR images by bicubic downsampling.

Usage:
    python scripts/generate_lr.py \
        --hr_dir custom_dataset/hr_valid_HR \
        --lr_dir custom_dataset/hr_valid_LR_x4 \
        --scale 4
"""
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def generate_lr(hr_dir: str, lr_dir: str, scale: int) -> None:
    hr_path = Path(hr_dir)
    lr_path = Path(lr_dir)
    lr_path.mkdir(parents=True, exist_ok=True)

    files = sorted(hr_path.glob("*.png")) + sorted(hr_path.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No PNG/JPG images found in {hr_dir}")

    for f in tqdm(files, desc=f"Generating LR (÷{scale})"):
        img = Image.open(f).convert("RGB")
        w, h = img.size
        lr = img.resize((w // scale, h // scale), Image.BICUBIC)
        lr.save(lr_path / f.name)

    print(f"Done — {len(files)} LR images saved to {lr_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr_dir", required=True, help="Folder with HR images")
    parser.add_argument("--lr_dir", required=True, help="Output folder for LR images")
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()
    generate_lr(args.hr_dir, args.lr_dir, args.scale)
