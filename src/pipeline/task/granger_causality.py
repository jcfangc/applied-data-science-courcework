from datetime import timedelta
from typing import AsyncGenerator, Dict, List, Set, Tuple

import numpy as np
from prefect.tasks import task_input_hash
from statsmodels.tsa.stattools import grangercausalitytests

from prefect import task

from ...definition.p_model.ess_divergence import (
    ESSSingleDivergence,
    ESSVariableDivergences,
)
from ...util.backoff import BackoffStrategy

# 创建退避策略实例 / Create a backoff strategy instance
backoff = BackoffStrategy()


class ESSCausalityCalculator:
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
        return sorted(
            divergences,
            key=lambda d: int("".join(filter(str.isdigit, d.round_from.name))),
        )

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
        sorted_divs = ESSCausalityCalculator.sort_divergences(
            ess_variable_divergences.divergences
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
    async def extract_unique_countries(
        generator: AsyncGenerator[ESSVariableDivergences, None],
    ) -> Set[str]:
        """
        从 ESSVariableDivergences 数据流中提取唯一 country 值。
        Extract unique country values from the ESSVariableDivergences data stream.

        :param generator: 异步生成器，逐步提供 ESSVariableDivergences 数据
                          Asynchronous generator yielding ESSVariableDivergences objects.
        :return: 唯一的国家名称集合 (Set[str])
                 A set of unique country names.
        """
        unique_countries = set()
        async for divergence in generator:
            unique_countries.add(divergence.country)
        return unique_countries

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
    ) -> Dict[Tuple[str, str], float]:
        """
        计算两个 ESSVariableDivergences 变量之间的因果关系（双向）。
        Compute the causal relationship between two ESSVariableDivergences variables (bidirectional).

        :param variable_a: 第一个变量的 JS 散度数据
                           The JS divergence data for the first variable.
        :param variable_b: 第二个变量的 JS 散度数据
                           The JS divergence data for the second variable.
        :param maxlag: 最大滞后期，默认值为 2 / Maximum lag, default is 2.
        :return: 一个包含双向因果关系的字典：
                 A dictionary containing bidirectional causal relationships:
                 {
                    (变量 A, 变量 B): 因果关系分数 (p-value),
                    (变量 B, 变量 A): 因果关系分数 (p-value)
                 }
        """
        # 提取 JS 散度数据作为时间序列 / Extract the JS divergence data as time series.
        js_a = np.array([d.js_divergence for d in variable_a.divergences])
        js_b = np.array([d.js_divergence for d in variable_b.divergences])

        # 确保两个变量的时间序列长度一致 / Ensure the time series lengths match.
        if len(js_a) != len(js_b):
            raise ValueError(
                f"变量 {variable_a.name} 和 {variable_b.name} 的时间序列长度不匹配！"
                f"Variable {variable_a.name} and {variable_b.name} have mismatched time series lengths!"
            )

        # 构造 Granger 因果分析输入数据 / Construct input data for Granger causality tests.
        granger_data = np.column_stack([js_a, js_b])

        # 执行 Granger 因果分析（最大滞后期设为 maxlag） / Run Granger causality tests with maximum lag = maxlag.
        granger_results = grangercausalitytests(
            granger_data, maxlag=maxlag, verbose=False
        )

        # 提取 p 值（选择滞后阶数为 2 的结果） / Extract p-values (using results for lag 2).
        a_to_b_p_value = granger_results[2][0]["ssr_ftest"][1]  # A → B
        b_to_a_p_value = granger_results[2][1]["ssr_ftest"][1]  # B → A

        return {
            (variable_a.name, variable_b.name): a_to_b_p_value,
            (variable_b.name, variable_a.name): b_to_a_p_value,
        }
