import asyncio
from typing import AsyncGenerator, Dict, List, Tuple

from pydantic import BaseModel, Field

from ..enum.causality_computation_status import CausalityComputationStatus


class CausalityAdjacencyMatrix(BaseModel):
    """维护因果关系的 3D 邻接表（支持异步并发安全）
    Maintains a 3D adjacency matrix for causality relationships (thread-safe).
    """

    adjacency_matrix: Dict[str, Dict[str, CausalityComputationStatus]] = Field(
        ...,
        description="变量间的因果计算状态邻接表 / Adjacency matrix for causality computation status",
    )

    # 🔐 线程/协程锁，保证并发安全
    _lock: asyncio.Lock = asyncio.Lock()

    @classmethod
    async def initialize(cls, variables: List[str]) -> "CausalityAdjacencyMatrix":
        """
        初始化邻接表，仅存储变量对的上三角部分，因果计算状态设为 `PENDING`。
        Initialize the adjacency matrix, storing only the upper triangular part,
        setting causality computation status to `PENDING` for all variable pairs.

        :param variables: 变量名称列表 / List of variable names
        :return: 初始化后的 `CausalityAdjacencyMatrix` 实例
                Initialized `CausalityAdjacencyMatrix` instance
        """
        # **先对变量列表进行排序**
        sorted_variables = sorted(variables)

        adjacency_matrix = {
            var1: {
                var2: CausalityComputationStatus.PENDING
                for var2 in sorted_variables[i + 1 :]
            }
            for i, var1 in enumerate(sorted_variables)
        }

        return cls(adjacency_matrix=adjacency_matrix)

    async def update_status(
        self, var1: str, var2: str, status: CausalityComputationStatus
    ) -> None:
        """
        **协程安全**：更新两个变量之间的因果计算状态，仅更新存储的上三角部分。
        **Thread-safe**: Update the causality computation status between two variables,
        ensuring only the upper triangular part is updated.

        :param var1: 变量 1 / First variable
        :param var2: 变量 2 / Second variable
        :param status: 更新后的状态 / Updated status
        """
        var1, var2 = sorted([var1, var2])  # 确保 var1 < var2 / Ensure var1 < var2

        async with self._lock:
            if var1 in self.adjacency_matrix and var2 in self.adjacency_matrix[var1]:
                self.adjacency_matrix[var1][var2] = status

    async def get_status(self, var1: str, var2: str) -> CausalityComputationStatus:
        """
        **协程安全**：获取两个变量之间的因果计算状态，仅检查存储的上三角部分。
        **Thread-safe**: Get the causality computation status between two variables,
        ensuring only the upper triangular part is checked.

        :param var1: 变量 1 / First variable
        :param var2: 变量 2 / Second variable
        :return: 计算状态 / Computation status
        """
        var1, var2 = sorted([var1, var2])  # 确保 var1 < var2 / Ensure var1 < var2

        async with self._lock:
            return self.adjacency_matrix.get(var1, {}).get(
                var2, CausalityComputationStatus.PENDING
            )

    async def iter_adjacency_entries(
        self,
    ) -> AsyncGenerator[Tuple[str, str, CausalityComputationStatus], None]:
        """
        **协程安全**：异步遍历 `adjacency_matrix`，逐个返回 `(var1, var2, status)`。\n
        **Thread-safe**: Asynchronous generator iterating over `adjacency_matrix`, yielding `(var1, var2, status)` one by one.

        :yield: `(var1, var2, status)` 元组，代表变量 `var1` 和 `var2` 之间的因果计算状态。\n
                `(var1, var2, status)` tuple representing the causality computation status between `var1` and `var2`.
        """
        async with self._lock:
            entries = [
                (var1, var2, status)
                for var1, sub_dict in self.adjacency_matrix.items()
                for var2, status in sub_dict.items()
            ]
        for entry in entries:
            yield entry
