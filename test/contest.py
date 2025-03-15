# conftest.py (放在项目根目录或 test 目录中)
import sys
from pathlib import Path

# 将项目根目录添加到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
