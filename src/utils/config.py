"""
src/utils/config.py
Config loader — merges base + task-specific YAML files.
Usage:
    cfg = load_config("configs/train_config.yaml")
"""

import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional


class DotDict(dict):
    """Dict subclass that supports dot-notation access: cfg.training.lr"""
    def __getattr__(self, key):
        try:
            val = self[key]
            return DotDict(val) if isinstance(val, dict) else val
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflicts."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(
    config_path: str,
    base_path: str = "configs/base_config.yaml",
    override: Optional[Dict[str, Any]] = None,
) -> DotDict:
    """
    Load config YAML, merging with base config first.

    Args:
        config_path: Path to task-specific config (e.g. configs/train_config.yaml)
        base_path:   Path to base defaults config
        override:    Extra key-value overrides applied last (e.g. from CLI flags)

    Returns:
        DotDict with merged config values
    """
    base_path = Path(base_path)
    config_path = Path(config_path)

    cfg = {}

    # 1. Load base defaults
    if base_path.exists():
        with open(base_path) as f:
            cfg = yaml.safe_load(f) or {}

    # 2. Merge task config on top
    if config_path.exists():
        with open(config_path) as f:
            task_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, task_cfg)
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # 3. Apply any runtime overrides
    if override:
        cfg = _deep_merge(cfg, override)

    return DotDict(cfg)


def save_config(cfg: dict, path: str) -> None:
    """Save config dict to YAML for experiment reproducibility."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(dict(cfg), f, default_flow_style=False, sort_keys=True)
    print(f"[Config] Saved to {path}")


def load_label_map(path: str) -> Dict[str, int]:
    """Load label_map.json: {gloss_word: integer_index}"""
    with open(path) as f:
        return json.load(f)
