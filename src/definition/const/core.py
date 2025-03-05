from pathlib import Path

# root
ROOT_DIR = Path(__file__).parent.parent.parent.parent

# 1
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR.parent / "data"
PREFECT_DIR = ROOT_DIR.parent / "prefect"

# 2
LOG_DIR = SRC_DIR / "log"
DIVERGENCE_DIR = DATA_DIR / "divergence"

# 3
JS_DIVERGENCE_DIR = DIVERGENCE_DIR / "jensen_shannon"
