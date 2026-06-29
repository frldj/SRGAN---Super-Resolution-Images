import argparse
import logging
from pathlib import Path

from config import load_config, merge_cli
from trainer import train, test, test_only


def _setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fmt = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(log_dir) / 'run.log'),
        ],
    )


def str2bool(v: str) -> bool:
    return v.lower() == 'true'


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='SRGAN — super-resolution training and evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--config', type=str, default='configs/default.yaml',
                   help='Path to YAML configuration file')

    # The arguments below override the YAML config when provided
    p.add_argument('--mode', type=str, choices=['train', 'test', 'test_only'])
    p.add_argument('--LR_path', type=str)
    p.add_argument('--GT_path', type=str)
    p.add_argument('--generator_path', type=str,
                   help='Checkpoint path for test/test_only, or fine-tuning resume')
    p.add_argument('--resume', type=str,
                   help='Alias: resume training from this checkpoint (sets fine_tuning=True)')
    p.add_argument('--fine_tuning', type=str2bool)
    p.add_argument('--batch_size', type=int)
    p.add_argument('--pre_train_epoch', type=int)
    p.add_argument('--fine_train_epoch', type=int)
    p.add_argument('--in_memory', type=str2bool)
    p.add_argument('--checkpoint_dir', type=str)
    p.add_argument('--log_dir', type=str)
    p.add_argument('--result_dir', type=str)
    return p


def main() -> None:
    cli = _build_parser().parse_args()
    args = load_config(cli.config)

    overrides = {k: v for k, v in vars(cli).items() if k != 'config' and v is not None}
    if cli.resume:
        overrides['generator_path'] = cli.resume
        overrides['fine_tuning'] = True

    args = merge_cli(args, overrides)

    _setup_logging(args.log_dir)
    logger = logging.getLogger(__name__)
    logger.info(f"Config: {cli.config} | Mode: {args.mode}")

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'test_only':
        test_only(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()
