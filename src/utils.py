import yaml
import os


def load_config(config_path=None):
    """
    Loads the configuration from a YAML file.

    Args:
        config_path (str, optional): Path to the configuration file. Defaults to None (uses default config.yaml).

    Returns:
        dict: The configuration dictionary.
    """
    if config_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
