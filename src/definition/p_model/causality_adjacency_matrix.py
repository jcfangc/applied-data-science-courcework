import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Tuple

import aiofiles
from pydantic import BaseModel, Field, PrivateAttr

from ..enum.causality_computation_status import CausalityComputationStatus


class CausalityAdjacencyMatrix(BaseModel):
    """
    维护因果关系的 3D 邻接表（支持异步并发安全），并支持持久化与恢复。
    Maintains a 3D adjacency matrix for causality relationships (thread-safe),
    with persistence and recovery support.
    """

    adjacency_matrix: Dict[str, Dict[str, CausalityComputationStatus]] = Field(
        ...,
        description="变量间的因果计算状态邻接表 / Adjacency matrix for causality computation status",
    )

    # 🔐 线程/协程锁，保证并发安全
    _lock: asyncio.Lock = asyncio.Lock()
    # 私有属性：更新计数器，每调用 20 次 update 进行一次持久化
    _update_count: int = PrivateAttr(default=0)
    # 私有属性：持久化目录，如果不为 None，则 update 时会自动保存状态
    _persist_path: Optional[Path] = PrivateAttr(default=None)

    @classmethod
    async def initialize(
        cls, variables: List[str], persist_path: Optional[Path] = None
    ) -> "CausalityAdjacencyMatrix":
        """
        初始化邻接表，仅存储变量对的上三角部分，因果计算状态设为 `PENDING`。
        如果传入持久化目录且其中有持久化文件，则恢复最新状态。

        Initialize the adjacency matrix, storing only the upper triangular part,
        setting causality computation status to `PENDING` for all variable pairs.
        If persist_path is provided and contains persisted data, recover from it.

        :param variables: 变量名称列表 / List of variable names
        :param persist_path: 持久化目录路径 / Directory path for persistence (optional)
        :return: 初始化后的 `CausalityAdjacencyMatrix` 实例
        """
        if persist_path is not None:
            # 如果目录存在且有持久化文件，则尝试恢复最新状态
            # If the directory exists and contains persisted files, attempt to recover the latest state
            latest_file: Optional[Path] = cls._get_latest_file(persist_path)
            if latest_file is not None and latest_file.exists():
                recovered_matrix = await cls.recover(latest_file)
                instance = cls(adjacency_matrix=recovered_matrix)
                instance._persist_path = persist_path
                return instance

        # 否则，正常初始化
        # Otherwise, initialize normally
        sorted_variables: List[str] = sorted(variables)
        adjacency_matrix: Dict[str, Dict[str, CausalityComputationStatus]] = {
            var1: {
                var2: CausalityComputationStatus.PENDING
                for var2 in sorted_variables[i + 1 :]
            }
            for i, var1 in enumerate(sorted_variables)
        }
        instance = cls(adjacency_matrix=adjacency_matrix)
        if persist_path is not None:
            instance._persist_path = persist_path
        return instance

    async def update_status(
        self, var1: str, var2: str, status: CausalityComputationStatus
    ) -> None:
        """
        **协程安全**：更新两个变量之间的因果计算状态，仅更新存储的上三角部分。
        每调用 20 次更新后自动持久化当前状态。

        **Thread-safe**: Update the causality computation status between two variables,
        ensuring only the upper triangular part is updated.
        Automatically persist the state every 20 updates.

        :param var1: 变量 1 / First variable
        :param var2: 变量 2 / Second variable
        :param status: 更新后的状态 / Updated status
        """
        # 确保 var1 < var2
        var1, var2 = sorted([var1, var2])

        async with self._lock:
            if var1 in self.adjacency_matrix and var2 in self.adjacency_matrix[var1]:
                self.adjacency_matrix[var1][var2] = status

            self._update_count += 1
            # 每 20 次更新后持久化（如果设置了持久化目录）
            if self._persist_path is not None and self._update_count >= 20:
                await self.persist()

    async def get_status(self, var1: str, var2: str) -> CausalityComputationStatus:
        """
        **协程安全**：获取两个变量之间的因果计算状态，仅检查存储的上三角部分。

        **Thread-safe**: Get the causality computation status between two variables,
        ensuring only the upper triangular part is checked.

        :param var1: 变量 1 / First variable
        :param var2: 变量 2 / Second variable
        :return: 计算状态 / Computation status
        """
        var1, var2 = sorted([var1, var2])
        return self.adjacency_matrix.get(var1, {}).get(
            var2, CausalityComputationStatus.PENDING
        )

    async def iter_adjacency_entries(
        self,
    ) -> AsyncGenerator[Tuple[str, str, CausalityComputationStatus], None]:
        """
        **协程安全**：异步遍历 `adjacency_matrix`，逐个返回 `(var1, var2, status)`。

        **Thread-safe**: Asynchronous generator iterating over `adjacency_matrix`,
        yielding `(var1, var2, status)` one by one.

        :yield: `(var1, var2, status)` 元组，代表变量 `var1` 和 `var2` 之间的因果计算状态。
        """
        # 读取操作直接遍历，不加锁提高并发性能
        for var1, sub_dict in self.adjacency_matrix.items():
            for var2, status in sub_dict.items():
                yield var1, var2, status

    async def persist(self) -> None:
        """
        将当前邻接矩阵状态持久化到指定目录中，文件名格式为“时间戳.json”。
        持久化后检查目录中 JSON 文件是否超过 20 个，若超过则删除最古早的文件。

        :raises ValueError: 如果未设置持久化目录
        """
        if self._persist_path is None:
            raise ValueError("持久化目录未设置 / Persistence directory not set.")

        # 确保目录存在
        self._persist_path.mkdir(parents=True, exist_ok=True)
        # 生成文件名，例如：20230310153045.json
        timestamp: str = datetime.now().strftime("%Y%m%d%H%M%S")
        file_path: Path = self._persist_path / f"{timestamp}.json"

        # 将枚举转换为其 name 保存
        persist_data: Dict[str, Dict[str, str]] = {
            var1: {var2: status.name for var2, status in sub_dict.items()}
            for var1, sub_dict in self.adjacency_matrix.items()
        }
        async with aiofiles.open(file_path, mode="w", encoding="utf-8") as f:
            await f.write(json.dumps(persist_data, indent=2))

        # 重置更新计数器
        # Reset the update counter
        self._update_count = 0

        # 检查目录中 JSON 文件数量，若超过 20 个则删除最古早的文件
        # Check the number of JSON files in the directory, delete the oldest file if more than 20
        json_files: list[Path] = sorted(self._persist_path.glob("*.json"))
        if len(json_files) > 20:
            # 计算需要删除的文件数
            num_to_delete: int = len(json_files) - 20
            for old_file in json_files[:num_to_delete]:
                old_file.unlink()

    @classmethod
    async def recover(
        cls, file_path: Path
    ) -> Dict[str, Dict[str, CausalityComputationStatus]]:
        """
        从指定的 JSON 文件中恢复邻接矩阵状态。假定文件中保存的是枚举的 name。

        Recover the adjacency matrix state from a JSON file.
        It is assumed that the enum values are stored as their names.

        :param file_path: 持久化文件路径 / The path to the persisted file.
        :return: 恢复后的邻接矩阵 / The recovered adjacency matrix.
        """
        async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
            content: str = await f.read()
            data: Dict[str, Dict[str, str]] = json.loads(content)
        # 将枚举 name 转换为枚举对象
        recovered_matrix: Dict[str, Dict[str, CausalityComputationStatus]] = {
            var1: {
                var2: CausalityComputationStatus[status_str]
                for var2, status_str in sub_dict.items()
            }
            for var1, sub_dict in data.items()
        }
        return recovered_matrix

    @staticmethod
    def _get_latest_file(directory: Path) -> Optional[Path]:
        """
        获取指定目录中最新的 JSON 文件，文件名格式为“时间戳.json”。

        Get the latest JSON file from the specified directory.
        Assumes the file name is in the format "timestamp.json".

        :param directory: 目录路径 / The directory path.
        :return: 最新的文件路径，如果不存在则返回 None / The latest file path, or None if not found.
        """
        if not directory.exists():
            return None
        json_files: List[Path] = sorted(directory.glob("*.json"))
        if not json_files:
            return None
        # 假定文件名按时间戳排序后，最后一个即最新
        return json_files[-1]
