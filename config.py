import yaml
from pathlib import Path
from types import SimpleNamespace


def load_config(config_path: str) -> SimpleNamespace:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg)


def merge_cli(cfg: SimpleNamespace, overrides: dict) -> SimpleNamespace:
    """Apply non-None CLI overrides on top of the YAML config."""
    cfg_dict = vars(cfg)
    cfg_dict.update({k: v for k, v in overrides.items() if v is not None})
    return SimpleNamespace(**cfg_dict)
