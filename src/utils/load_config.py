import yaml
from pathlib import Path


def load_config(config_path=None):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str or Path, optional): Path to the config file.
                                             If None, defaults to 'config.yaml' in the project root.

    Returns:
        dict: The configuration dictionary.
    """
    if config_path is None:
        # Assuming this file is in src/utils/, the project root is two levels up
        # However, let's be more robust. If we are running from root, it might be just 'config.yaml'
        # Let's try to find the project root relative to this file.
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        config_path = project_root / "config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config
