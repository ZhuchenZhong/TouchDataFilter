import configparser
import torch
from pathlib import Path

# Create config object
config = configparser.ConfigParser()

# Set current values from your code
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_ROOT = PROJECT_ROOT / "model"
SENSITIVITY_ROOT = PROJECT_ROOT / "result" / "sensitivity"
PREDICTED_ROOT = PROJECT_ROOT / "result" / "predicted"
CONFIG_ROOT = PROJECT_ROOT / "config"

if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    batch_size = min(128, max(32, int(gpu_mem * 4)))
    GPU_SUPPORT = True
else:
    batch_size = 32
    GPU_SUPPORT = False


config['PATHS'] = {
    'PROJECT_ROOT': str(PROJECT_ROOT),
    'DATA_ROOT': str(DATA_ROOT),
    'MODEL_ROOT': str(MODEL_ROOT),
    'SENSITIVITY_ROOT': str(SENSITIVITY_ROOT),
    'PREDICTED_ROOT': str(PREDICTED_ROOT)
}

config['GPU'] = {
    'GPU_SUPPORT': str(GPU_SUPPORT),
    'BATCH_SIZE': str(batch_size)
}

# Write to file
with open(CONFIG_ROOT / 'settings.ini', 'w') as f:
    config.write(f)
