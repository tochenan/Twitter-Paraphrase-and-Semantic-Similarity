from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
SYSTEMOUTPUTS_DIR = PROJECT_ROOT / "systemoutputs"
SRC_DIR = PROJECT_ROOT / "src"

TRAIN_DATA_PATH = DATA_DIR / "train.data"
TEST_DATA_PATH = DATA_DIR / "test.data"
TEST_LABEL_PATH = DATA_DIR / "test.label"
