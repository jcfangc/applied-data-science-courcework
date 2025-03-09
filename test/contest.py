# conftest.py (放在项目根目录或 test 目录中)
import sys
from pathlib import Path

from src.definition.const.prefect import LOCAL_STORAGE

# 将项目根目录添加到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 持久化本地存储块到服务器
LOCAL_STORAGE.save(name="local-storage", overwrite=True)
