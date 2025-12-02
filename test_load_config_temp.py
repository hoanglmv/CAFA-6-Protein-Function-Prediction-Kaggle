from src.utils.load_config import load_config
import sys
from pathlib import Path

# Add src to python path so we can import if running from root
sys.path.append(str(Path.cwd()))

try:
    config = load_config()
    print("Successfully loaded config:")
    print(config)
except Exception as e:
    print(f"Failed to load config: {e}")
