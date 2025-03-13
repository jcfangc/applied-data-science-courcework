from pathlib import Path

# root
ROOT_DIR = Path(__file__).parent.parent.parent.parent

# 1
SRC_DIR = ROOT_DIR / "src"

# 2
LOG_DIR = SRC_DIR / "log"
CACHE_DIR = SRC_DIR / "cache"
RESULT_DIR = SRC_DIR / "result"

# 3
DIVERGENCE_CACHE_DIR = CACHE_DIR / "divergence"
CAUSALITY_RESULT_DIR = RESULT_DIR / "causality"
