import yaml
from pathlib import Path

def load_config(config_path="config.yaml"):
    """Config" YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_feature_names(config):
    """Feature names based on config"""
    features = config['variables'][:-1]  # All except target
    if config.get('add_temporal_features', False):
        features.extend(['year', 'month_sin', 'month_cos'])
    return features