from pathlib import Path

# root
ROOT_DIR = Path(__file__).parent.parent.parent.parent

# 1
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR.parent / "data"
PREFECT_DIR = ROOT_DIR.parent / "prefect"
HELP_DIR = ROOT_DIR.parent / "help"

# 2
LOG_DIR = SRC_DIR / "log"
DIVERGENCE_DIR = DATA_DIR / "divergence"
CAUSALITY_DIR = DATA_DIR / "causality"
GENERATOR_CACHE_DIR = HELP_DIR / "generator_cache"

# 3
JS_DIVERGENCE_DIR = DIVERGENCE_DIR / "jensen_shannon"
DEFAULT_CACHE_FILE = GENERATOR_CACHE_DIR / "default_cached_generator.jsonl"
