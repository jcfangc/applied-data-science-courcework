from datetime import timedelta
from typing import AsyncGenerator, Dict, Set, Tuple

from loguru import logger

from prefect import task
from prefect.tasks import task_input_hash

from ...definition.enum.causality_computation_status import CausalityComputationStatus
from ...definition.p_model.causality_adjacency_matrix import CausalityAdjacencyMatrix
from ...definition.p_model.ess_divergence import (
    ESSVariableDivergences,
)
from ...util.backoff import BackoffStrategy

# 创建退避策略实例
backoff = BackoffStrategy()


class AdjacencyMatrixTask:
    """
    因果邻接表任务类，包含构建因果邻接表和提取邻接表条目的任务。
    Causality adjacency matrix task class, which includes tasks to build causality adjacency matrices
    """

    @staticmethod
    @task(
        persist_result=True,
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
        cache_key_fn=lambda *_: "build_causality_adjacency_matrices",
        cache_expiration=timedelta(hours=1),
    )
    async def build_causality_adjacency_matrices(
        generator: AsyncGenerator[ESSVariableDivergences, None],
    ) -> Dict[str, CausalityAdjacencyMatrix]:
        """
        组装因果邻接表，返回每个国家对应的 CausalityAdjacencyMatrix。
        Build causality adjacency matrices and return a dictionary mapping
        each country to its corresponding `CausalityAdjacencyMatrix`.

        :param generator: 异步生成器，逐步提供 ESSVariableDivergences 数据
                        Asynchronous generator yielding ESSVariableDivergences objects.
        :return: 每个国家对应的因果邻接表 (Dict[str, CausalityAdjacencyMatrix])
                A dictionary mapping each country to its causality adjacency matrix.
        """

        data_list = [item async for item in generator]

        # 按国家分类变量 / Group variables by country.
        country_to_variables: Dict[str, Set] = {}
        for item in data_list:
            if item.country not in country_to_variables:
                country_to_variables[item.country] = set()
            country_to_variables[item.country].add(item.name)

        # 构造邻接表 / Build adjacency matrices.
        adjacency_matrices = {
            country: CausalityAdjacencyMatrix.initialize(list(variables))
            for country, variables in country_to_variables.items()
        }

        logger.info(
            f"成功为 {len(adjacency_matrices)} 个国家构建因果邻接表。\n"
            f"Successfully built causality adjacency matrices for {len(adjacency_matrices)} countries."
        )

        return adjacency_matrices

    @staticmethod
    @task(
        persist_result=True,
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
        cache_key_fn=task_input_hash,
        cache_expiration=timedelta(hours=1),
    )
    async def extract_adjacency_entries(
        adjacency_matrices: Dict[str, CausalityAdjacencyMatrix],
    ) -> AsyncGenerator[Tuple[str, str, str, CausalityComputationStatus], None]:
        """
        **异步生成器**，遍历多个 `CausalityAdjacencyMatrix`，逐个返回 `(country, var1, var2, status)`。
        Asynchronous generator iterating over multiple `CausalityAdjacencyMatrix`,
        yielding `(country, var1, var2, status)` one by one.

        :param adjacency_matrices: 每个国家对应的因果邻接表。
                                Dictionary mapping each country to its causality adjacency matrix.
        :yield: `(country, var1, var2, status)`，代表 `var1` 和 `var2` 在 `country` 的因果计算状态。
                `(country, var1, var2, status)`, representing the causality computation status
                between `var1` and `var2` in a given `country`.
        """

        for country, matrix in adjacency_matrices.items():
            async for var1, var2, status in matrix.iter_adjacency_entries():
                yield (country, var1, var2, status)
