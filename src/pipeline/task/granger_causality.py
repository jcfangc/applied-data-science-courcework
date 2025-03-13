import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger
from prefect import task
from prefect.tasks import task_input_hash
from statsmodels.tsa.stattools import grangercausalitytests
from tqdm.asyncio import tqdm

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
            f"开始对国家 {ess_variable_divergences.country} 的变量 {ess_variable_divergences.name} 进行散度排序\n"
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

    @task(
        persist_result=True,
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
        cache_key_fn=task_input_hash,
        cache_expiration=timedelta(hours=1),
    )
    def bidirectional_granger_test(
        js_a: np.ndarray, js_b: np.ndarray, maxlag: int
    ) -> Tuple[float, float]:
        """
        对两个时间序列分别进行 Granger 因果检验，返回双向的 p-value。

        :param js_a: 第一个时间序列数据（例如变量 A）
                     First time series data (e.g., variable A).
        :param js_b: 第二个时间序列数据（例如变量 B）
                     Second time series data (e.g., variable B).
        :param maxlag: 最大滞后阶数
                       Maximum lag order.
        :return: Tuple (p_value_a_causes_b, p_value_b_causes_a)
        """
        # 测试 js_a 是否 Granger-causes js_b：
        # 构造输入数据时：第一列作为被解释变量（js_b），第二列作为解释变量（js_a）
        granger_data_a2b = np.column_stack([js_b, js_a])
        results_a2b = grangercausalitytests(granger_data_a2b, maxlag=maxlag)
        p_value_a2b = results_a2b[maxlag][0]["ssr_ftest"][1]

        # 测试 js_b 是否 Granger-causes js_a：
        granger_data_b2a = np.column_stack([js_a, js_b])
        results_b2a = grangercausalitytests(granger_data_b2a, maxlag=maxlag)
        p_value_b2a = results_b2a[maxlag][0]["ssr_ftest"][1]

        return p_value_a2b, p_value_b2a

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

        try:
            a_to_b_p_value, b_to_a_p_value = (
                ESSCausalityCalculatorTask.bidirectional_granger_test(
                    js_a, js_b, maxlag
                )
            )
        except Exception as e:
            logger.error(f"Granger 因果分析失败 / Granger causality test failed: {e}")
            raise e

        logger.info(
            f"因果关系计算完成 / Causality computation completed:\n"
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

    @task(
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
    )
    async def process_causality(
        country: str,
        var1: str,
        var2: str,
        semaphore: asyncio.Semaphore,
        js_divergence_cache: Dict[Tuple[str, str], ESSVariableDivergences],
        adjacency_matrices: Dict[str, CausalityAdjacencyMatrix],
        maxlag: int,
    ) -> Tuple[ESSCausalityResult, ESSCausalityResult]:
        """并行计算因果关系的子任务"""
        async with semaphore:
            variable_a = js_divergence_cache.get((country, var1))
            variable_b = js_divergence_cache.get((country, var2))

            logger.info(f"开始计算因果关系：{country} {var1} {var2}")

            if not variable_a or not variable_b:
                logger.warning(f"缺失 {country} 的变量数据: {var1}, {var2}，跳过。")
                return None  # 跳过计算

            # 更新邻接表状态
            await adjacency_matrices[country].update_status(
                var1, var2, CausalityComputationStatus.IN_PROGRESS
            )

            # 后台线程计算（避免阻塞事件循环）
            result_a2b, result_b2a = await asyncio.to_thread(
                ESSCausalityCalculatorTask.compute_causal_relationship,
                variable_a=variable_a,
                variable_b=variable_b,
                maxlag=maxlag,
            )

            result_a2b: ESSCausalityResult
            result_b2a: ESSCausalityResult
            logger.info(
                f"因果关系计算完成：{result_a2b.country} {result_a2b.causality_from} → {result_a2b.causality_to}，"
                f"p-value = {result_a2b.p_value}"
                f"因果关系计算完成：{result_b2a.country} {result_b2a.causality_from} → {result_b2a.causality_to}，"
                f"p-value = {result_b2a.p_value}"
            )

            # 更新邻接表状态
            await adjacency_matrices[country].update_status(
                var1, var2, CausalityComputationStatus.SUCCESS
            )

            return result_a2b, result_b2a

    @task(
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
    )
    async def load_existing_results(output_csv: Path) -> Set[Tuple[str, str, str]]:
        """
        加载已计算的因果关系结果。
        Load previously computed causality results.

        :param output_csv: 输出 CSV 文件路径。
                            Output CSV file path.
        :return: 已计算结果的集合，格式为 (country, var1, var2)。
                Set of previously computed results, formatted as (country, var1, var2).
        """
        existing_results: Set[
            Tuple[str, str, str]
        ] = await ReadWriteTask.load_existing_causality_results(output_csv)
        logger.info(
            f"✅ 已加载 {len(existing_results)} 个计算过的因果关系，避免重复计算.\n"
            f"✅ Successfully loaded {len(existing_results)} previously computed causality relationships, avoiding duplicate computations."
        )
        return existing_results

    @task(
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
    )
    def build_js_divergence_cache(
        divergences_list: List[ESSVariableDivergences],
    ) -> Dict[Tuple[str, str], ESSVariableDivergences]:
        """
        构建 JS 散度数据的缓存字典。
        Construct a cache dictionary for JS divergence data.

        :param divergences_list: ESSVariableDivergences 列表。
                                List of ESSVariableDivergences.
        :return: 以 (country, variable_name) 为键的 JS 散度数据字典。
                JS divergence data dictionary with (country, variable_name) as key.
        """
        cache: Dict[Tuple[str, str], ESSVariableDivergences] = {
            (js_data.country, js_data.name): js_data for js_data in divergences_list
        }
        logger.info(
            f"成功加载 {len(cache)} 个 JS 散度数据 / Successfully loaded {len(cache)} JS divergence data."
        )
        return cache

    @task(
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
    )
    async def calculate_total_entries(
        adjacency_matrices: Dict[str, CausalityAdjacencyMatrix],
    ) -> int:
        """
        计算所有邻接矩阵中条目的总数。
        Calculate the total number of entries in all adjacency matrices.

        :param adjacency_matrices: country -> adjacency matrix 的字典。
                                    Dictionary mapping country to adjacency matrix.
        :return: 邻接条目的总数。
                Total number of adjacency entries.
        """
        return await asyncio.to_thread(
            lambda: sum(
                sum(len(sub_dict) for sub_dict in matrix.adjacency_matrix.values())
                for matrix in adjacency_matrices.values()
            )
        )

    @task(
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
    )
    async def create_causality_tasks(
        adjacency_entries_generator: AsyncGenerator[
            Tuple[str, str, str, CausalityComputationStatus], None
        ],
        total_entries: int,
        existing_results: Set[Tuple[str, str, str]],
        semaphore: asyncio.Semaphore,
        js_divergence_cache: Dict[Tuple[str, str], ESSVariableDivergences],
        adjacency_matrices: Dict[str, CausalityAdjacencyMatrix],
        maxlag: int,
    ) -> List[asyncio.Task[Any]]:
        """
        遍历邻接条目流，创建待计算因果关系的异步任务列表。
        Iterate over the adjacency entry stream and create a list of asynchronous tasks to compute causality relationships.

        :param adjacency_entries_generator: 包含 (country, var1, var2, status) 的异步生成器。
                                            Asynchronous generator containing (country, var1, var2, status).
        :param total_entries: 邻接矩阵总条目数，用于进度显示。
                                Total number of entries in the adjacency matrices, used for progress display.
        :param existing_results: 已计算结果的集合。
                                Set of previously computed results.
        :param semaphore: 限制并行计算任务的信号量。
                            Semaphore to limit parallel computation tasks.
        :param js_divergence_cache: JS 散度数据缓存。
                                    JS divergence data cache.
        :param adjacency_matrices: 因果邻接矩阵字典。
                                    Causality adjacency matrix dictionary.
        :param maxlag: Granger 滞后阶数。
                        Granger lag order.
        :return: 异步任务列表。
                Asynchronous task list.
        """
        tasks: List[asyncio.Task[Any]] = []
        async for country, var1, var2, status in tqdm(
            adjacency_entries_generator,
            total=total_entries,
            desc="Processing causality",
        ):
            if (country, var1, var2) in existing_results:
                logger.info(
                    f"⏭️ 结果已存在，跳过计算：{country} {var1} & {var2}\n"
                    f"⏭️ Result already exists, skipping computation: {country} {var1} & {var2}"
                )
                continue

            if status == CausalityComputationStatus.PENDING:
                task = asyncio.create_task(
                    ESSCausalityCalculatorTask.process_causality(
                        country=country,
                        var1=var1,
                        var2=var2,
                        semaphore=semaphore,
                        js_divergence_cache=js_divergence_cache,
                        adjacency_matrices=adjacency_matrices,
                        maxlag=maxlag,
                    )
                )
                tasks.append(task)
        return tasks

    @task(
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
    )
    async def compute_causal_relationship_from_adjacency(
        adjacency_matrices: Dict[str, CausalityAdjacencyMatrix],
        adjacency_entries_generator: AsyncGenerator[
            Tuple[str, str, str, CausalityComputationStatus], None
        ],
        divergences_list: List[ESSVariableDivergences],
        output_csv: Path,
        maxlag: int = 2,
        batch_size: int = 10,  # 定义每个持久化批次的大小
    ) -> None:
        """
        从邻接表提取变量对，并计算因果关系，避免重复计算，
        计算结果将分批持久化到 CSV 文件中。
        Extract variable pairs from the adjacency matrices and compute causality relationships,
        avoiding duplicate computations. Persist the results in batches to a CSV file.

        :param adjacency_matrices: 因果邻接表字典，包含 country -> adjacency matrix。
                                    Causality adjacency matrix dictionary, containing country -> adjacency matrix.
        :param adjacency_entries_generator: 邻接表条目流，包含 (country, var1, var2, status)。
                                            Adjacency matrix entry stream, containing (country, var1, var2, status).
        :param divergences_list: ESSVariableDivergences 列表，包含所有变量的 JS 散度数据。
                                List of ESSVariableDivergences containing JS divergence data for all variables.
        :param output_csv: 输出 CSV 文件路径，用于保存计算结果。
                            Output CSV file path to save the computation results.
        :param maxlag: Granger 滞后阶数，默认 2。
                        Granger lag order, default is 2.
        :param batch_size: 每个持久化批次处理的结果数量。
                            Number of results processed in each persistence batch.
        :return: None
        """
        if not divergences_list:
            logger.warning(
                "❌ divergences_list 为空，因果关系计算无法进行！\n"
                "❌ divergences_list is empty, causality computation cannot proceed!"
            )
            return

        # 加载已计算结果
        # Load existing results
        existing_results: Set[
            Tuple[str, str, str]
        ] = await ESSCausalityCalculatorTask.load_existing_results(output_csv)

        # 加载已计算结果及构建 JS 散度缓存
        # Load existing results and build JS divergence cache
        js_divergence_cache: Dict[Tuple[str, str], ESSVariableDivergences] = (
            ESSCausalityCalculatorTask.build_js_divergence_cache(divergences_list)
        )

        # 计算邻接矩阵条目总数
        # Calculate the total number of entries in the adjacency matrices
        total_entries: int = await ESSCausalityCalculatorTask.calculate_total_entries(
            adjacency_matrices
        )
        logger.info(
            f"邻接表总条目数: {total_entries}\n"
            f"Total number of adjacency entries: {total_entries}"
        )

        # 限制并行计算的最大任务数
        # Limit the maximum number of parallel tasks
        semaphore: asyncio.Semaphore = asyncio.Semaphore(10)

        # 创建待执行的计算任务
        # Create tasks to be executed
        tasks: List[
            asyncio.Task[Any]
        ] = await ESSCausalityCalculatorTask.create_causality_tasks(
            adjacency_entries_generator=adjacency_entries_generator,
            total_entries=total_entries,
            existing_results=existing_results,
            semaphore=semaphore,
            js_divergence_cache=js_divergence_cache,
            adjacency_matrices=adjacency_matrices,
            maxlag=maxlag,
        )

        # 使用 asyncio.as_completed 实时处理任务结果，并按批次持久化保存
        # Process task results in real-time using asyncio.as_completed and persist in batches
        batch: List[ESSCausalityResult] = []
        for completed_task in asyncio.as_completed(tasks):
            result: Optional[
                Tuple[ESSCausalityResult, ESSCausalityResult]
            ] = await completed_task
            if result:
                # 添加返回的两个 ESSCausalityResult 到当前批次
                # Add the two returned ESSCausalityResults to the current batch
                batch.extend(result)
            if len(batch) >= batch_size:
                await ReadWriteTask.save_causality_results_to_csv(batch, output_csv)
                batch = []  # 重置批次 / Reset batch

        # 持久化剩余不足一个批次的结果
        # Persist the remaining results that do not form a full batch
        if batch:
            await ReadWriteTask.save_causality_results_to_csv(batch, output_csv)
