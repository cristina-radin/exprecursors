import yaml
from pathlib import Path


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_feature_names(config):
    """Input variable names (no target, no land_mask)."""
    return list(config["variables"])
