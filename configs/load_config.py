"""Load the default config plus one Mean Flow YAML recipe.

Usage examples:
  --config configs/load_config.py:distill_b16
  --config configs/load_config.py:configs/eval_distill_config_b16.yml

The string after the colon is deliberately an argument. There is no special
`remote_run_config.yml` path baked into the training code.
"""

import os

import yaml

from configs.default import get_config as get_default_config


_CONFIG_ALIASES = {
    "distill": "distill_config_b16.yml",
    "distill_b16": "distill_config_b16.yml",
    "eval": "eval_distill_config_b16.yml",
    "eval_distill": "eval_distill_config_b16.yml",
    "eval_distill_b16": "eval_distill_config_b16.yml",
}


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_config_file(config_name):
    if not config_name:
        raise ValueError("Config name must be provided after configs/load_config.py:")

    root = _repo_root()
    configs_dir = os.path.join(root, "configs")
    aliased = _CONFIG_ALIASES.get(config_name, config_name)

    candidates = []
    if os.path.isabs(aliased):
        candidates.append(aliased)
    else:
        candidates.append(os.path.join(root, aliased))
        candidates.append(os.path.join(configs_dir, aliased))
        if not aliased.endswith((".yml", ".yaml")):
            candidates.append(os.path.join(configs_dir, f"{aliased}.yml"))
            candidates.append(os.path.join(configs_dir, f"{aliased}_config.yml"))

    for path in candidates:
        if os.path.exists(path):
            return path

    tried = "\n  ".join(candidates)
    raise FileNotFoundError(f"Could not resolve config {config_name!r}. Tried:\n  {tried}")


def _merge_dict(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and key in dst and hasattr(dst[key], "update"):
            _merge_dict(dst[key], value)
        else:
            dst[key] = value


def get_config(config_name):
    config_file = _resolve_config_file(config_name)
    with open(config_file) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    config = get_default_config()
    _merge_dict(config, config_dict)
    config.config_file = config_file
    return config
