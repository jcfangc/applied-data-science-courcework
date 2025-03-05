from prefect.filesystems import LocalFileSystem

from .core import PREFECT_DIR


MAX_RETRIES = 5
LOCAL_STORAGE = LocalFileSystem(basepath=PREFECT_DIR)
