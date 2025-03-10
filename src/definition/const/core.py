from pathlib import Path

# root
ROOT_DIR = Path(__file__).parent.parent.parent.parent

# 1
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR.parent / "data"
HELP_DIR = ROOT_DIR.parent / "help"

# 2
LOG_DIR = SRC_DIR / "log"
CAUSALITY_DIR = DATA_DIR / "causality"
GENERATOR_CACHE_DIR = HELP_DIR / "generator_cache"
ADJACENCY_MATRIX_DIR = HELP_DIR / "adjacency_matrix"

# 3
DEFAULT_CACHE_FILE = GENERATOR_CACHE_DIR / "default_cached_generator.jsonl"
