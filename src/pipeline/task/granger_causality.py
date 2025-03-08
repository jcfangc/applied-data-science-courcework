from datetime import timedelta
from pathlib import Path
from typing import AsyncGenerator, Awaitable, Callable, Dict, List, Tuple

import numpy as np
from loguru import logger
from statsmodels.tsa.stattools import grangercausalitytests

from prefect import task
from prefect.tasks import task_input_hash

from ...definition.enum.causality_computation_status import CausalityComputationStatus
from ...definition.p_model.causality_adjacency_matrix import CausalityAdjacencyMatrix
from ...definition.p_model.ess_causality import ESSCausalityResult
from ...definition.p_model.ess_divergence import (
    ESSSingleDivergence,
    ESSVariableDivergences,
)
from ...util.backoff import BackoffStrategy
from .read_write import ReadWriteTask

# 创建退避策略实例 / Create a backoff strategy instance
backoff = BackoffStrategy()


class ESSCausalityCalculatorTask:
    """
    因果计算工具类，包含辅助任务，如对 ESSSingleDivergence 列表进行排序，
    以及对 ESSVariableDivergences 对象中的散度列表进行排序。
    ESS Causality Calculator utility class, which includes helper tasks such as
    sorting a list of ESSSingleDivergences and sorting the divergence list within an ESSVariableDivergences object.
    """

    @staticmethod
    @task(
        cache_key_fn=task_input_hash,
        cache_expiration=timedelta(hours=1),
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
        persist_result=True,
    )
    def sort_divergences(
        divergences: List[ESSSingleDivergence],
    ) -> List[ESSSingleDivergence]:
        """
        对 ESSSingleDivergence 列表按照 round_from 中的数字顺序进行排序。
        Sort the list of ESSSingleDivergence objects based on the numeric part in round_from.
        假设 Round 的名称格式为 "ESS<number>"。
        Assumes that the Round name is formatted as "ESS<number>".

        :param divergences: 未排序的 ESSSingleDivergence 列表
                            Unsorted list of ESSSingleDivergence objects.
        :return: 按照 round_from 排序后的列表
                List sorted by round_from.
        """

        # 任务输入检查 / Input validation
        if not divergences:
            logger.warning(
                "输入列表为空，返回空列表 / Input list is empty, returning empty list."
            )
            return []

        if not all(isinstance(d, ESSSingleDivergence) for d in divergences):
            logger.error(
                "输入数据格式错误，必须是 ESSSingleDivergence 列表 / Input data format error, must be a list of ESSSingleDivergence objects."
            )
            raise ValueError("输入数据格式错误 / Input data format error.")

        # 执行排序 / Perform sorting
        sorted_divs = sorted(
            divergences,
            key=lambda d: int("".join(filter(str.isdigit, d.round_from.name))),
        )

        logger.info(
            f"成功排序 {len(divergences)} 个 JS 散度数据项 / Successfully sorted {len(divergences)} JS divergence items."
        )

        return sorted_divs

    @staticmethod
    @task(
        cache_key_fn=task_input_hash,
        cache_expiration=timedelta(hours=1),
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
        persist_result=True,
    )
    def sort_variable_divergence(
        ess_variable_divergences: ESSVariableDivergences,
    ) -> ESSVariableDivergences:
        """
        将 ESSVariableDivergences 中的散度列表按照 round_from 排序后返回新的 ESSVariableDivergences 对象。
        Sort the divergence list within an ESSVariableDivergences object based on round_from,
        and return a new ESSVariableDivergences object.

        :param ess_variable_divergences: 原始 ESSVariableDivergences 对象
                                        Original ESSVariableDivergences object.
        :return: 排序后的 ESSVariableDivergences 对象
                Sorted ESSVariableDivergences object.
        """

        logger.info(
            f"开始对国家 {ess_variable_divergences.country} 的变量 {ess_variable_divergences.name} 进行散度排序 / "
            f"Starting to sort divergences for variable {ess_variable_divergences.name} from country {ess_variable_divergences.country}."
        )

        # 任务输入检查 / Input validation
        if not ess_variable_divergences.divergences:
            logger.warning(
                f"变量 {ess_variable_divergences.name} 的散度列表为空，直接返回原对象 / "
                f"Divergence list for variable {ess_variable_divergences.name} is empty, returning original object."
            )
            return ess_variable_divergences

        sorted_divs = ESSCausalityCalculatorTask.sort_divergences(
            ess_variable_divergences.divergences
        )

        logger.info(
            f"变量 {ess_variable_divergences.name} 的散度排序完成，共处理 {len(sorted_divs)} 个散度值 / "
            f"Sorting of divergences for variable {ess_variable_divergences.name} completed, processed {len(sorted_divs)} divergence values."
        )

        return ESSVariableDivergences(
            name=ess_variable_divergences.name,
            country=ess_variable_divergences.country,
            divergences=sorted_divs,
        )

    @staticmethod
    @task(
        persist_result=True,
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
        cache_key_fn=task_input_hash,
        cache_expiration=timedelta(hours=1),
    )
    def compute_causal_relationship(
        variable_a: ESSVariableDivergences,
        variable_b: ESSVariableDivergences,
        maxlag: int = 2,
    ) -> Tuple[ESSCausalityResult, ESSCausalityResult]:
        """
        计算两个 ESSVariableDivergences 变量之间的因果关系（双向）。
        Compute the causal relationship between two ESSVariableDivergences variables (bidirectional).

        :param variable_a: 第一个变量的 JS 散度数据
                           The JS divergence data for the first variable.
        :param variable_b: 第二个变量的 JS 散度数据
                           The JS divergence data for the second variable.
        :param maxlag: 最大滞后期，默认值为 2 / Maximum lag, default is 2.
        :return: 一个包含双向因果关系的列表：
                 A list containing bidirectional causal relationships:
                 [
                    ESSCausalityResult(country, 变量 A, 变量 B, 因果关系分数 (p-value)),
                    ESSCausalityResult(country, 变量 B, 变量 A, 因果关系分数 (p-value))
                 ]
        """
        logger.info(
            f"开始计算因果关系：{variable_a.name} ↔ {variable_b.name} / "
            f"Starting causality computation: {variable_a.name} ↔ {variable_b.name}"
        )

        # 确保两个变量属于同一个国家 / Ensure both variables belong to the same country.
        if variable_a.country != variable_b.country:
            raise ValueError(
                f"变量 {variable_a.name} ({variable_a.country}) 和 {variable_b.name} ({variable_b.country}) "
                f"不属于同一个国家！\n"
                f"Variables {variable_a.name} ({variable_a.country}) and {variable_b.name} ({variable_b.country}) "
                f"do not belong to the same country!"
            )

        # 调整顺序，确保两个散度列表顺序一致 / Adjust order to ensure the two divergence lists match.
        variable_a = ESSCausalityCalculatorTask.sort_variable_divergence(variable_a)
        variable_b = ESSCausalityCalculatorTask.sort_variable_divergence(variable_b)

        # 提取 JS 散度数据作为时间序列 / Extract JS divergence data as time series.
        js_a = np.array([d.js_divergence for d in variable_a.divergences])
        js_b = np.array([d.js_divergence for d in variable_b.divergences])

        # 确保两个变量的时间序列长度一致 / Ensure the time series lengths match.
        if len(js_a) != len(js_b):
            raise ValueError(
                f"变量 {variable_a.name} 和 {variable_b.name} 的时间序列长度不匹配！\n"
                f"Variable {variable_a.name} and {variable_b.name} have mismatched time series lengths!"
            )

        logger.info(
            f"Granger 因果分析输入数据构造完成，样本长度: {len(js_a)}\n"
            f"Granger causality test input data constructed, sample length: {len(js_a)}"
        )

        # 构造 Granger 因果分析输入数据 / Construct input data for Granger causality tests.
        granger_data = np.column_stack([js_a, js_b])

        # 执行 Granger 因果分析（最大滞后期设为 maxlag） / Run Granger causality tests with maximum lag = maxlag.
        granger_results = grangercausalitytests(
            granger_data, maxlag=maxlag, verbose=False
        )

        # 提取 p 值（选择滞后阶数为 2 的结果） / Extract p-values (using results for lag 2).
        a_to_b_p_value = granger_results[maxlag][0]["ssr_ftest"][1]  # A → B
        b_to_a_p_value = granger_results[maxlag][1]["ssr_ftest"][1]  # B → A

        logger.info(
            f"因果关系计算完成:\n"
            f"[{variable_a.country}]: {variable_a.name} → {variable_b.name}: p-value = {a_to_b_p_value}\n"
            f"[{variable_a.country}]: {variable_b.name} → {variable_a.name}: p-value = {b_to_a_p_value}\n"
            f"Causality computation complete:\n"
            f"[{variable_a.country}]: {variable_a.name} → {variable_b.name}: p-value = {a_to_b_p_value}\n"
            f"[{variable_a.country}]: {variable_b.name} → {variable_a.name}: p-value = {b_to_a_p_value}"
        )

        return (
            ESSCausalityResult(
                country=variable_a.country,
                causality_from=variable_a.name,
                causality_to=variable_b.name,
                p_value=a_to_b_p_value,
            ),
            ESSCausalityResult(
                country=variable_a.country,
                causality_from=variable_b.name,
                causality_to=variable_a.name,
                p_value=b_to_a_p_value,
            ),
        )

    @staticmethod
    @task(
        persist_result=False,  # 仅流式返回，由 Flow 负责存储
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
        cache_key_fn=task_input_hash,
        cache_expiration=timedelta(hours=1),
    )
    async def compute_causal_relationship_from_adjacency(
        adjacency_matrices: Dict[str, CausalityAdjacencyMatrix],
        adjacency_entries_generator: AsyncGenerator[
            Tuple[str, str, str, CausalityComputationStatus], None
        ],
        js_divergence_gen_cache_file: str | Path,
        deserialize_fn: Callable[[dict], Awaitable[ESSVariableDivergences]],
        maxlag: int = 2,
    ) -> AsyncGenerator[ESSCausalityResult, None]:
        """
        从邻接表提取变量对，并计算因果关系，避免重复计算。
        Extracts variable pairs from adjacency matrix and computes causality, avoiding redundant computations.

        :param adjacency_matrices: 因果邻接表字典，包含 `country -> adjacency matrix`。
                                   Dictionary mapping `country -> adjacency matrix`.
        :param adjacency_entries_generator: 邻接表条目流，包含 `(country, var1, var2, status)`。
                                           Adjacency matrix entry stream containing `(country, var1, var2, status)`.
        :param js_divergence_cache_file: 缓存文件路径，用于重新加载JS散度数据生成器。
                                         Cache file path used to reload JS divergence data generator.
        :param deserialize_fn: 异步反序列化函数。
                               Asynchronous deserialization function.
        :param maxlag: Granger 滞后阶数，默认 `2`。
                       Granger test max lag, default is `2`.
        :yield: `ESSCausalityResult`，包含 `country, var1, var2, p_value`。
                Yields `ESSCausalityResult`, containing `country, var1, var2, p_value`.
        """

        async for country, var1, var2, status in adjacency_entries_generator:
            if status != CausalityComputationStatus.PENDING:
                continue  # 只处理 PENDING 状态的变量对

            logger.info(
                f"提取 {country} 变量 {var1} 和 {var2} 进行因果分析。\n"
                f"Extracting {var1} and {var2} from {country} for causality analysis."
            )

            # 每次都从缓存重新加载生成器，确保可重复消费
            js_divergence_generator: AsyncGenerator[ESSVariableDivergences, None] = (
                ReadWriteTask.load_async_generator_from_cache(
                    deserialize_fn=deserialize_fn,
                    cache_file=js_divergence_gen_cache_file,
                )
            )

            variable_a, variable_b = None, None
            async for js_data in js_divergence_generator:
                if js_data.country == country and js_data.name == var1:
                    variable_a = js_data
                elif js_data.country == country and js_data.name == var2:
                    variable_b = js_data
                if variable_a and variable_b:
                    break  # 找到两个变量，停止遍历

            if not variable_a or not variable_b:
                logger.warning(
                    f"缺失 {country} 的变量数据: {var1}, {var2}，跳过。\n"
                    f"Missing variable data for {country}: {var1}, {var2}, skipping."
                )
                continue

            # 更新邻接表状态 / Update adjacency matrix status
            adjacency_matrices[country].update_status(
                var1, var2, CausalityComputationStatus.IN_PROGRESS
            )

            # 调用 `compute_causal_relationship` 计算因果关系 / Call `compute_causal_relationship` to compute causality
            result_a, result_b = ESSCausalityCalculatorTask.compute_causal_relationship(
                variable_a=variable_a,
                variable_b=variable_b,
                maxlag=maxlag,
            )

            logger.info(
                f"计算完成 {result_a.causality_from} → {result_a.causality_to} (p={result_a.p_value})\n"
                f"计算完成 {result_b.causality_from} → {result_b.causality_to} (p={result_b.p_value})\n"
                f"Completed causality computation {result_a.causality_from} → {result_a.causality_to} (p={result_a.p_value})\n"
                f"Completed causality computation {result_b.causality_from} → {result_b.causality_to} (p={result_b.p_value})"
            )

            # 更新邻接表状态 / Update adjacency matrix status
            adjacency_matrices[country].update_status(
                var1, var2, CausalityComputationStatus.SUCCESS
            )

            yield result_a
            yield result_b
