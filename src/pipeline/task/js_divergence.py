import asyncio
from datetime import timedelta
from typing import AsyncGenerator, Dict, List, Optional

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

# 创建退避策略实例
backoff = BackoffStrategy()


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
        logger.info("Completed compute_js_divergence: result = {result}", result=result)
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
    async def compute_js_batch(
        generator: AsyncGenerator[ESSVariableData, None],
        target_name: Optional[str] = None,
        target_country: Optional[str] = None,
        batch_size: int = 10,
    ) -> AsyncGenerator[ESSVariableDivergences, None]:
        """
        计算多个变量的 JS 散度，并根据 `target_name` 和 `target_country` 进行筛选。
        Compute JS divergence for multiple variables, filtering based on `target_name` and `target_country`.

        :param generator: 异步生成器，逐步提供 ESSVariableData 实例
                        Asynchronous generator providing instances of ESSVariableData.
        :param target_name: 目标变量名称，可选，若提供则仅返回该变量的结果
                            Target variable name, optional. If provided, only return results for this variable.
        :param target_country: 目标国家，可选，若提供则仅返回该国家的数据
                            Target country, optional. If provided, only return results for this country.
        :param batch_size: 批次大小，用于并行处理数据
                            Batch size for parallel processing of data.
        :yield: 满足筛选条件的 ESSVariableDivergences 对象
                Yields `ESSVariableDivergences` objects matching the filtering criteria.
        """

        # 内部函数：并行处理一批数据
        async def process_batch(
            batch: list[ESSVariableData],
        ) -> list[ESSVariableDivergences]:
            results = await asyncio.gather(
                *[
                    asyncio.to_thread(
                        ESSDivergenceCalculatorTask.compute_js_series,
                        data.distributions,
                    )
                    for data in batch
                ]
            )
            return [
                ESSVariableDivergences(
                    name=data.name, country=data.country, divergences=js_series
                )
                for data, js_series in zip(batch, results)
            ]

        batch: list[ESSVariableData] = []
        async for ess_data in generator:
            # 筛选符合条件的数据
            if target_name and ess_data.name != target_name:
                continue
            if target_country and ess_data.country != target_country:
                continue

            batch.append(ess_data)
            if len(batch) >= batch_size:
                divergences_list = await process_batch(batch)
                for divergence in divergences_list:
                    yield divergence
                batch = []  # 清空当前批次

        # 处理剩余不足一个批次的数据
        if batch:
            divergences_list = await process_batch(batch)
            for divergence in divergences_list:
                yield divergence
