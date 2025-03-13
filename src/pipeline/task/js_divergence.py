import asyncio
from datetime import timedelta
from pathlib import Path
from typing import AsyncGenerator, Dict, List, TypeVar

import polars as pl
from loguru import logger
from prefect import task
from prefect.tasks import task_input_hash
from scipy.spatial.distance import jensenshannon

from ...definition.enum.round import Round
from ...definition.p_model.ess_divergence import (
    ESSSingleDivergence,
    ESSVariableDivergences,
)
from ...definition.p_model.ess_variable_data import ESSVariableData
from ...util.backoff import BackoffStrategy
from .read_write import ReadWriteTask

# 创建退避策略实例
backoff = BackoffStrategy()

# 类型变量
T = TypeVar("T")


class ESSDivergenceCalculatorTask:
    """
    计算 ESS 数据集中，不同轮次之间，同一国家的变量分布散度。
    Calculate the distribution divergence of variables for the same country
    across different rounds in the ESS dataset.
    """

    @task(
        cache_key_fn=task_input_hash,
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
        cache_expiration=timedelta(hours=1),
        persist_result=True,
    )
    def compute_js_divergence(p: pl.Series, q: pl.Series) -> float:
        """
        计算两个 `polars.Series` 概率分布的 Jensen-Shannon 散度。
        Compute the Jensen-Shannon divergence between two `polars.Series` probability distributions.

        :param p: 第一个概率分布 (`pl.Series`)
                  First probability distribution (`pl.Series`)
        :param q: 第二个概率分布 (`pl.Series`)
                  Second probability distribution (`pl.Series`)
        :return: JS 散度
                 JS divergence
        """
        if not isinstance(p, pl.Series) or not isinstance(q, pl.Series):
            logger.error(
                "输入错误：输入必须是 polars.Series / Inputs must be polars.Series"
            )
            raise ValueError("输入必须是 polars.Series / Inputs must be polars.Series")
        if p.len() != q.len():
            logger.error(
                "输入错误：两个输入的长度不一致 / Both inputs must have the same length"
            )
            raise ValueError(
                "两个输入的长度必须相同 / Both inputs must have the same length"
            )
        if p.len() == 0:
            logger.error("输入错误：输入数组为空 / Input arrays cannot be empty")
            raise ValueError("输入数组不能为空 / Input arrays cannot be empty")

        p_np = p.to_numpy()
        q_np = q.to_numpy()
        result = jensenshannon(p_np + 1e-10, q_np + 1e-10) ** 2
        logger.info(
            f"计算 JS 散度完成: 结果 = {result}\n"
            f"Completed compute_js_divergence: result = {result}"
        )
        return result

    @task(
        cache_key_fn=task_input_hash,
        cache_expiration=timedelta(hours=1),
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
        persist_result=True,
    )
    def compute_js_series(
        distributions: Dict[Round, pl.Series],
    ) -> List[ESSSingleDivergence]:
        """
        计算有序概率分布集合中的相邻分布的 JS 散度。
        Compute the JS divergence between adjacent distributions in an ordered
        set of probability distributions.

        :param distributions: 变量在不同轮次的概率分布 (Dict[Round, pl.Series])
                              Probability distributions of variables across different rounds (Dict[Round, pl.Series])
        :return: List[ESSDivergence]，包含每个相邻轮次之间的 JS 散度
                 List[ESSDivergence], containing JS divergence between each adjacent round.
        """
        logger.info(
            "开始计算相邻轮次的 JS 散度\nStarting computation of JS divergence between adjacent rounds."
        )

        # 参数检查 / Parameter validation
        if not isinstance(distributions, dict) or not all(
            isinstance(k, Round) and isinstance(v, pl.Series)
            for k, v in distributions.items()
        ):
            logger.error(
                "输入错误: 期望的格式为 {Round: polars.Series}\n"
                "Input error: Expected format {Round: polars.Series}"
            )
            raise ValueError(
                "输入必须是 {Round: polars.Series} 形式的字典 / Input must be a dictionary in the form {Round: polars.Series}"
            )

        if len(distributions) < 2:
            logger.error(
                "数据错误: 至少需要 2 轮数据进行计算\n"
                "Data error: At least 2 rounds of data are required for computation."
            )
            raise ValueError(
                "必须至少包含两个轮次的数据才能计算 JS 散度 / At least two rounds of data are required to compute JS divergence"
            )

        # 按 Round 的顺序排序 / Sort by Round order
        sorted_rounds = sorted(
            distributions.keys(),
            key=lambda r: int("".join(filter(str.isdigit, r.name))),
        )
        sorted_distributions = [distributions[r] for r in sorted_rounds]

        logger.info(
            f"数据轮次排序完成: {sorted_rounds}\nData rounds sorted: {sorted_rounds}"
        )

        # 计算相邻轮次的 JS 散度 / Compute JS divergence between adjacent rounds
        js_results = []
        for i in range(len(sorted_distributions) - 1):
            round_from = sorted_rounds[i]
            round_to = sorted_rounds[i + 1]
            logger.info(
                f"计算 {round_from} → {round_to} 的 JS 散度\n"
                f"Computing JS divergence from {round_from} → {round_to}"
            )

            js_divergence = ESSDivergenceCalculatorTask.compute_js_divergence(
                sorted_distributions[i], sorted_distributions[i + 1]
            )

            logger.info(
                f"{round_from} → {round_to} 计算完成, JS 散度值: {js_divergence}\n"
                f"Computation completed for {round_from} → {round_to}, JS divergence: {js_divergence}"
            )

            js_results.append(
                ESSSingleDivergence(
                    round_from=round_from,
                    round_to=round_to,
                    js_divergence=js_divergence,
                )
            )

        logger.info(
            "所有相邻轮次的 JS 散度计算完成\n"
            "All adjacent rounds' JS divergence computations completed."
        )

        return js_results

    @task(
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
    )
    async def load_cached_data(
        cache_file: str | Path,
    ) -> Dict[tuple[str, str], ESSVariableDivergences]:
        """
        从 JSONL 缓存文件加载已计算的数据，并转换为字典索引。
        Load computed data from a JSONL cache file and convert to a dictionary index.

        :param cache_file: 缓存文件路径
                            Path of cache file
        :return: 以 (变量名, 国家) 为键的已缓存数据字典
                Dictionary of cached data with (variable name, country) as key
        """
        cache_file = Path(cache_file)
        cached_data: Dict[tuple[str, str], ESSVariableDivergences] = {}

        if cache_file.exists():
            logger.info(
                f"正在从 {cache_file} 加载数据...\n"
                f"Loading cached data from {cache_file}..."
            )
            async for cached_item in ReadWriteTask.load_async_generator_from_cache(
                deserialize_fn=lambda data: asyncio.to_thread(
                    ESSVariableDivergences.model_validate, data
                ),
                cache_file=cache_file,
            ):
                cached_item: ESSVariableDivergences
                cache_key = (cached_item.name, cached_item.country)
                cached_data[cache_key] = cached_item

            logger.info(
                f"从 {cache_file} 加载了 {len(cached_data)} 条缓存结果\n"
                f"Loaded {len(cached_data)} cached results from {cache_file}"
            )
        else:
            logger.warning(
                f"缓存文件未找到: {cache_file}\nCache file not found: {cache_file}"
            )

        return cached_data

    @task(
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
    )
    def should_skip_computation(
        ess_data: ESSVariableData,
        cached_data: Dict[tuple[str, str], ESSVariableDivergences],
    ) -> bool:
        """
        判断是否需要跳过某个数据项的计算。
        Determine whether to skip computation for a data item.

        :param ess_data: 当前数据项
                        Current data item
        :param cached_data: 已缓存数据
                            Cached data
        :return: 若该数据项已存在缓存或不符合筛选条件，则返回 True
                If the data item is already cached or does not meet the filtering conditions, return True
        """
        cache_key = (ess_data.name, ess_data.country)
        if cache_key in cached_data:
            logger.info(
                f"跳过 ({ess_data.name}, {ess_data.country}) 的计算 - 已存在缓存。\n"
                f"Skipping computation for ({ess_data.name}, {ess_data.country}) - already cached."
            )
            return True
        return False

    @task(
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
    )
    async def compute_js_for_batch(
        batch: list[ESSVariableData],
    ) -> list[ESSVariableDivergences]:
        """
        并行计算一个批次的数据的 JS 散度。
        Parallel computation of JS divergence for a batch of data.

        :param batch: 需要计算的批次数据
                    batch of data to compute
        :return: 计算得到的 `ESSVariableDivergences` 列表
                    List of computed `ESSVariableDivergences`
        """
        logger.info(
            f"为批次中的 {len(batch)} 个数据计算 JS 散度...\n"
            f"Computing JS divergence for batch of {len(batch)} items..."
        )

        results = await asyncio.gather(
            *[
                asyncio.to_thread(
                    ESSDivergenceCalculatorTask.compute_js_series,
                    data.distributions,
                )
                for data in batch
            ]
        )

        logger.info(
            f"为批次中的 {len(batch)} 个数据计算 JS 散度完成。\n"
            f"Completed JS divergence computation for batch of {len(batch)} items."
        )

        return [
            ESSVariableDivergences(
                name=data.name, country=data.country, divergences=js_series
            )
            for data, js_series in zip(batch, results)
        ]

    @task(
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
    )
    async def store_results_to_cache(
        results: list[ESSVariableDivergences], cache_file: str | Path
    ):
        """
        将计算出的 JS 散度结果存入缓存文件（追加模式）。
        Store computed JS divergence results to cache file (append mode).


        :param results: 需要存储的数据列表
                        List of data to store
        :param cache_file: 缓存文件路径
                            Cache file path
        """
        if not results:
            logger.info("没有新结果存入缓存。\nNo new results to store in cache.")
            return

        logger.info(
            f"将 {len(results)} 条新结果存入缓存: {cache_file}\n"
            f"Storing {len(results)} new results to cache: {cache_file}"
        )

        async def async_generator_from_list(
            data_list: list[T],
        ) -> AsyncGenerator[T, None]:
            """
            将普通列表转换为异步生成器
            Convert a normal list to an async generator
            """
            for item in data_list:
                yield item

        try:
            await ReadWriteTask.cache_async_generator(
                generator=async_generator_from_list(
                    results
                ),  # 直接迭代写入 / Write directly by iteration
                serialize_fn=lambda data: asyncio.to_thread(
                    ESSVariableDivergences.serialize, data
                ),
                cache_file=cache_file,
            )
        except Exception as e:
            logger.error(
                f"存储数据时出现错误: {e}\nError occurred while storing data: {e}"
            )
            raise e

        logger.info(
            f"成功将 {len(results)} 条新结果存入 {cache_file}\n"
            f"Successfully stored {len(results)} new results to {cache_file}"
        )

    @task(
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
    )
    async def compute_js_batch(
        generator: AsyncGenerator[ESSVariableData, None],
        cache_file: str | Path,
        batch_size: int = 10,
    ) -> List[ESSVariableDivergences]:
        """
        计算多个变量的 JS 散度，并根据 `target_name` 和 `target_country` 进行筛选。
        Compute JS divergence for multiple variables, and filter by `target_name` and `target_country`.
        计算前检查缓存，避免重复计算，计算完成后增量存储，并最终返回 **完整数据列表**（非生成器）。
        Check cache before computation to avoid duplicates, store results incrementally after computation.

        :param generator: 异步生成器，逐步提供 ESSVariableData 实例
                            Async generator providing ESSVariableData instances one by one
        :param cache_file: 缓存文件路径
                            Cache file path
        :param batch_size: 批次大小，用于并行处理数据
                            Batch size for parallel processing of data
        :return: 计算完成的 `ESSVariableDivergences` 对象列表
                    List of computed `ESSVariableDivergences` objects
        """
        logger.info(
            "开始计算 JS 散度批次数据...\nStarting JS divergence batch computation..."
        )

        # Step 1: 先加载缓存中的数据
        # Step 1: Load cached data first
        cached_data: Dict[
            tuple[str, str], ESSVariableDivergences
        ] = await ESSDivergenceCalculatorTask.load_cached_data(cache_file)

        # Step 2: 计算新数据（避免重复计算）
        # Step 2: Compute new data (avoid duplicates)
        new_results: List[ESSVariableDivergences] = []
        batch: list[ESSVariableData] = []

        async for ess_data in generator:
            if ESSDivergenceCalculatorTask.should_skip_computation(
                ess_data, cached_data
            ):
                continue  # 该数据已存在缓存，跳过 / Data already cached, skip

            batch.append(ess_data)
            if len(batch) >= batch_size:
                divergences_list = (
                    await ESSDivergenceCalculatorTask.compute_js_for_batch(batch)
                )

                # Step 3: 计算完后增量存储
                # Step 3: Incremental storage after computation
                await ESSDivergenceCalculatorTask.store_results_to_cache(
                    divergences_list, cache_file
                )

                new_results.extend(divergences_list)
                batch = []  # 清空当前批次 / Clear current batch

        # 处理剩余数据
        # Process remaining data
        if batch:
            divergences_list = await ESSDivergenceCalculatorTask.compute_js_for_batch(
                batch
            )
            await ESSDivergenceCalculatorTask.store_results_to_cache(
                divergences_list, cache_file
            )
            new_results.extend(divergences_list)

        # Step 4: 返回完整数据（缓存 + 新计算）
        # Step 4: Return complete data (cached + newly computed)
        logger.info(
            f"JS 散度批次计算完成。总计结果: {len(cached_data) + len(new_results)}\n"
            f"JS divergence batch computation completed. Total results: {len(cached_data) + len(new_results)}"
        )

        return list(cached_data.values()) + new_results
