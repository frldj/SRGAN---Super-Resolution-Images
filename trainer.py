import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from dataset import SRDataset, TestOnlyDataset, RandomCrop, RandomAugment
from losses import TVLoss, PerceptualLoss
from srgan_model import Generator, Discriminator
from vgg19 import VGG19

logger = logging.getLogger(__name__)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _setup_dirs(args: SimpleNamespace) -> None:
    for d in [args.checkpoint_dir, args.log_dir, args.result_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)


def _validate(args: SimpleNamespace, require_gt: bool = True) -> None:
    if not Path(args.LR_path).exists():
        raise FileNotFoundError(f"LR path not found: {args.LR_path}")
    if require_gt and not Path(args.GT_path).exists():
        raise FileNotFoundError(f"GT path not found: {args.GT_path}")
    if args.fine_tuning and not args.generator_path:
        raise ValueError("--generator_path is required when fine_tuning=True")
    if args.mode in ('test', 'test_only') and not args.generator_path:
        raise ValueError("--generator_path is required for test/test_only modes")


def save_checkpoint(path: str, generator: nn.Module, epoch: int, phase: str,
                    g_optim: optim.Optimizer = None,
                    discriminator: nn.Module = None,
                    d_optim: optim.Optimizer = None) -> None:
    state = {
        'epoch': epoch,
        'phase': phase,
        'generator': generator.state_dict(),
    }
    if g_optim:
        state['g_optim'] = g_optim.state_dict()
    if discriminator:
        state['discriminator'] = discriminator.state_dict()
    if d_optim:
        state['d_optim'] = d_optim.state_dict()
    torch.save(state, path)
    logger.info(f"Checkpoint saved → {path}")


_LEGACY_KEY_MAP = {
    'conv01.': 'head.',
    'conv02.': 'body_tail.',
}


def _remap_state_dict(sd: dict) -> dict:
    """Translate pre-refactor key names to current architecture names."""
    remapped = {}
    for k, v in sd.items():
        for old, new in _LEGACY_KEY_MAP.items():
            if k.startswith(old):
                k = new + k[len(old):]
                break
        remapped[k] = v
    return remapped


def load_checkpoint(path: str, generator: nn.Module, device: torch.device,
                    g_optim: optim.Optimizer = None,
                    discriminator: nn.Module = None,
                    d_optim: optim.Optimizer = None) -> tuple[int, str]:
    """Load checkpoint and return (epoch, phase). Handles both old and new formats."""
    state = torch.load(path, map_location=device, weights_only=True)

    # New format: dict with 'generator' key + metadata
    if isinstance(state, dict) and 'generator' in state and isinstance(state['generator'], dict):
        generator.load_state_dict(_remap_state_dict(state['generator']))
        if g_optim and 'g_optim' in state:
            g_optim.load_state_dict(state['g_optim'])
        if discriminator and 'discriminator' in state:
            discriminator.load_state_dict(state['discriminator'])
        if d_optim and 'd_optim' in state:
            d_optim.load_state_dict(state['d_optim'])
        epoch, phase = state.get('epoch', 0), state.get('phase', 'pre')
    else:
        # Old format: plain state dict saved directly
        generator.load_state_dict(_remap_state_dict(state))
        epoch, phase = 0, 'pre'

    logger.info(f"Checkpoint loaded ← {path} (epoch={epoch}, phase={phase})")
    return epoch, phase


def _build_generator(args: SimpleNamespace) -> Generator:
    return Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=args.res_num, scale=args.scale)


# ── Train ─────────────────────────────────────────────────────────────────────

def train(args: SimpleNamespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    _setup_dirs(args)
    _validate(args, require_gt=True)

    transform = transforms.Compose([RandomCrop(args.scale, args.patch_size), RandomAugment()])
    dataset = SRDataset(GT_path=args.GT_path, LR_path=args.LR_path,
                        in_memory=args.in_memory, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    logger.info(f"Dataset: {len(dataset)} images | {len(loader)} batches/epoch")

    generator = _build_generator(args).to(device)
    generator.train()

    l2_loss = nn.MSELoss()
    g_optim = optim.Adam(generator.parameters(), lr=args.lr)

    start_epoch, phase = 0, 'pre'
    if args.fine_tuning and args.generator_path:
        start_epoch, phase = load_checkpoint(args.generator_path, generator, device, g_optim=g_optim)

    writer = SummaryWriter(log_dir=args.log_dir)

    # ── Phase 1: SRResNet pre-training (L2) ──────────────────────────────────
    if phase == 'pre':
        logger.info(f"[Pre-train] Epochs {start_epoch + 1} → {args.pre_train_epoch}")

        for epoch in range(start_epoch + 1, args.pre_train_epoch + 1):
            epoch_loss = 0.0
            pbar = tqdm(loader, desc=f"Pre-train [{epoch}/{args.pre_train_epoch}]", leave=False)

            for batch in pbar:
                gt = batch['GT'].to(device)
                lr_img = batch['LR'].to(device)

                output, _ = generator(lr_img)
                loss = l2_loss(gt, output)

                g_optim.zero_grad()
                loss.backward()
                g_optim.step()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.6f}")

            avg_loss = epoch_loss / len(loader)
            writer.add_scalar('pre_train/L2_loss', avg_loss, epoch)

            if epoch % args.log_interval == 0:
                logger.info(f"[Pre-train] epoch {epoch:5d} | L2: {avg_loss:.6f}")

            if epoch % args.pre_checkpoint_interval == 0:
                ckpt = Path(args.checkpoint_dir) / f"srresnet_epoch{epoch:04d}.pt"
                save_checkpoint(str(ckpt), generator, epoch, phase='pre', g_optim=g_optim)

        # Always save at the end of pre-training
        ckpt = Path(args.checkpoint_dir) / f"srresnet_epoch{args.pre_train_epoch:04d}.pt"
        save_checkpoint(str(ckpt), generator, args.pre_train_epoch, phase='pre', g_optim=g_optim)

        start_epoch = 0  # reset counter for fine-tune phase

    # ── Phase 2: SRGAN fine-tuning (perceptual + adversarial) ────────────────
    logger.info(f"[Fine-tune] Epochs {start_epoch + 1} → {args.fine_train_epoch}")

    vgg_net = VGG19().to(device).eval()

    discriminator = Discriminator(patch_size=args.patch_size * args.scale).to(device)
    discriminator.train()

    d_optim = optim.Adam(discriminator.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(g_optim, step_size=args.lr_step_size, gamma=args.lr_gamma)

    vgg_loss = PerceptualLoss(vgg_net).to(device)
    cross_ent = nn.BCELoss()
    tv_loss = TVLoss()

    for epoch in range(start_epoch + 1, args.fine_train_epoch + 1):
        g_epoch_loss = d_epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Fine-tune [{epoch}/{args.fine_train_epoch}]", leave=False)

        for batch in pbar:
            gt = batch['GT'].to(device)
            lr_img = batch['LR'].to(device)
            n = gt.size(0)  # actual batch size (last batch may be smaller)
            real_label = torch.ones(n, 1, device=device)
            fake_label = torch.zeros(n, 1, device=device)

            # Discriminator step
            with torch.no_grad():
                fake, _ = generator(lr_img)
            d_loss = (
                cross_ent(discriminator(gt), real_label)
                + cross_ent(discriminator(fake), fake_label)
            )
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            # Generator step
            output, _ = generator(lr_img)
            fake_prob = discriminator(output)

            _percep_loss, hr_feat, sr_feat = vgg_loss(
                (gt + 1.0) / 2.0, (output + 1.0) / 2.0, layer=args.feat_layer
            )
            g_loss = (
                args.vgg_rescale_coeff * _percep_loss
                + args.adv_coeff * cross_ent(fake_prob, real_label)
                + args.tv_loss_coeff * tv_loss(args.vgg_rescale_coeff * (hr_feat - sr_feat) ** 2)
                + args.L2_coeff * l2_loss(output, gt)
            )
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

            g_epoch_loss += g_loss.item()
            d_epoch_loss += d_loss.item()
            pbar.set_postfix(G=f"{g_loss.item():.4f}", D=f"{d_loss.item():.4f}")

        scheduler.step()

        avg_g = g_epoch_loss / len(loader)
        avg_d = d_epoch_loss / len(loader)
        writer.add_scalar('fine_tune/G_loss', avg_g, epoch)
        writer.add_scalar('fine_tune/D_loss', avg_d, epoch)

        if epoch % args.log_interval == 0:
            logger.info(f"[Fine-tune] epoch {epoch:5d} | G: {avg_g:.6f} | D: {avg_d:.6f}")

        if epoch % args.checkpoint_interval == 0:
            ckpt = Path(args.checkpoint_dir) / f"srgan_epoch{epoch:04d}.pt"
            save_checkpoint(str(ckpt), generator, epoch, phase='fine',
                            g_optim=g_optim, discriminator=discriminator, d_optim=d_optim)

    # Always save at the end of fine-tuning
    ckpt = Path(args.checkpoint_dir) / f"srgan_epoch{args.fine_train_epoch:04d}.pt"
    save_checkpoint(str(ckpt), generator, args.fine_train_epoch, phase='fine',
                    g_optim=g_optim, discriminator=discriminator, d_optim=d_optim)

    writer.close()
    logger.info("Training complete.")


# ── Test (with GT, computes PSNR) ─────────────────────────────────────────────

def test(args: SimpleNamespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _setup_dirs(args)
    _validate(args, require_gt=True)

    dataset = SRDataset(GT_path=args.GT_path, LR_path=args.LR_path, in_memory=False, transform=None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    generator = _build_generator(args)
    load_checkpoint(args.generator_path, generator, device)
    generator = generator.to(device).eval()

    psnr_list = []
    result_txt = Path(args.result_dir) / 'psnr_results.txt'

    with torch.no_grad(), open(result_txt, 'w') as f:
        for i, batch in enumerate(tqdm(loader, desc='Testing')):
            gt = batch['GT'].to(device)
            lr_img = batch['LR'].to(device)

            _, _, h, w = lr_img.size()
            gt = gt[:, :, :h * args.scale, :w * args.scale]

            output, _ = generator(lr_img)
            output = np.clip(output[0].cpu().numpy(), -1.0, 1.0)
            gt_np = gt[0].cpu().numpy()

            output = ((output + 1.0) / 2.0).transpose(1, 2, 0)
            gt_np = ((gt_np + 1.0) / 2.0).transpose(1, 2, 0)

            y_out = rgb2ycbcr(output)[args.scale:-args.scale, args.scale:-args.scale, :1]
            y_gt = rgb2ycbcr(gt_np)[args.scale:-args.scale, args.scale:-args.scale, :1]

            psnr = peak_signal_noise_ratio(y_out / 255.0, y_gt / 255.0, data_range=1.0)
            psnr_list.append(psnr)
            f.write(f"img {i:04d} | psnr: {psnr:.4f}\n")

            Image.fromarray((output * 255.0).astype(np.uint8)).save(
                Path(args.result_dir) / f"res_{i:04d}.png"
            )

        avg = np.mean(psnr_list)
        f.write(f"\navg psnr: {avg:.4f}")
        logger.info(f"Test complete | avg PSNR: {avg:.4f} dB | results → {args.result_dir}")


# ── Test-only (no GT, inference only) ────────────────────────────────────────

def test_only(args: SimpleNamespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _setup_dirs(args)
    _validate(args, require_gt=False)

    dataset = TestOnlyDataset(LR_path=args.LR_path, in_memory=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    generator = _build_generator(args)
    load_checkpoint(args.generator_path, generator, device)
    generator = generator.to(device).eval()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc='Inference')):
            lr_img = batch['LR'].to(device)
            output, _ = generator(lr_img)
            output = ((output[0].cpu().numpy() + 1.0) / 2.0).transpose(1, 2, 0)
            Image.fromarray((output * 255.0).astype(np.uint8)).save(
                Path(args.result_dir) / f"res_{i:04d}.png"
            )

    logger.info(f"Inference complete | {len(dataset)} images → {args.result_dir}")
