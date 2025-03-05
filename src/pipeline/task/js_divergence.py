from datetime import timedelta
from typing import AsyncGenerator, Dict, List

import numpy as np
from prefect.tasks import task_input_hash
from scipy.spatial.distance import jensenshannon

from prefect import task

from ...definition.const.prefect import LOCAL_STORAGE
from ...definition.enum.round import Round
from ...definition.p_model.ess_divergence import (
    ESSSingleDivergence,
    ESSVariableDivergence,
)
from ...definition.p_model.ess_variable_data import ESSVariableData
from ...util.backoff import BackoffStrategy

# 创建退避策略实例
backoff = BackoffStrategy()


class ESSDivergenceCalculator:
    """
    计算 ESS 数据集中，不同轮次之间，同一国家的变量分布散度。
    Calculate the distribution divergence of variables for the same country
    across different rounds in the ESS dataset.
    """

    @staticmethod
    @task(
        cache_key_fn=task_input_hash,
        retries=5,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
        cache_expiration=timedelta(hours=1),
        persist_result=True,
        result_storage=LOCAL_STORAGE,
    )
    def compute_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        计算两个概率分布之间的 Jensen-Shannon 散度。
        Compute the Jensen-Shannon divergence between two probability distributions.

        :param p: 第一个概率分布 (ndarray)
                  First probability distribution (ndarray)
        :param q: 第二个概率分布 (ndarray)
                  Second probability distribution (ndarray)
        :return: JS 散度
                 JS divergence
        """
        if not isinstance(p, np.ndarray) or not isinstance(q, np.ndarray):
            raise ValueError("输入必须是 numpy.ndarray / Inputs must be numpy.ndarray")
        if p.shape != q.shape:
            raise ValueError(
                "两个输入的形状必须相同 / Both inputs must have the same shape"
            )
        if len(p) == 0:
            raise ValueError("输入数组不能为空 / Input arrays cannot be empty")

        return (
            jensenshannon(p + 1e-10, q + 1e-10) ** 2
        )  # 还原 JS 散度 / Restore JS divergence

    @staticmethod
    @task(
        cache_key_fn=task_input_hash,
        cache_expiration=timedelta(hours=1),
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
        persist_result=True,
        result_storage=LOCAL_STORAGE,
    )
    def compute_js_series(
        distributions: Dict[Round, np.ndarray]
    ) -> List[ESSSingleDivergence]:
        """
        计算有序概率分布集合中的相邻分布的 JS 散度。
        Compute the JS divergence between adjacent distributions in an ordered
        set of probability distributions.

        :param distributions: 变量在不同轮次的概率分布 (Dict[Round, np.ndarray])
                              Probability distributions of variables across different rounds (Dict[Round, np.ndarray])
        :return: List[ESSDivergence]，包含每个相邻轮次之间的 JS 散度
                 List[ESSDivergence], containing JS divergence between each adjacent round.
        """
        if not isinstance(distributions, dict) or not all(
            isinstance(k, Round) and isinstance(v, np.ndarray)
            for k, v in distributions.items()
        ):
            raise ValueError(
                "输入必须是 {Round: np.ndarray} 形式的字典 / Input must be a dictionary in the form {Round: np.ndarray}"
            )

        if len(distributions) < 2:
            raise ValueError(
                "必须至少包含两个轮次的数据才能计算 JS 散度 / At least two rounds of data are required to compute JS divergence"
            )

        # 按 Round 的顺序排序 / Sort by Round order
        sorted_rounds = sorted(distributions.keys(), key=lambda r: r.value)
        sorted_distributions = [distributions[r] for r in sorted_rounds]

        # 计算相邻轮次的 JS 散度 / Compute JS divergence between adjacent rounds
        js_results = [
            ESSSingleDivergence(
                round_from=sorted_rounds[i],
                round_to=sorted_rounds[i + 1],
                js_divergence=ESSDivergenceCalculator.compute_js_divergence(
                    sorted_distributions[i], sorted_distributions[i + 1]
                ),
            )
            for i in range(len(sorted_distributions) - 1)
        ]

        return js_results

    @staticmethod
    @task(
        persist_result=False,  # 流式返回不直接持久化，由 Flow 负责消费
        # Stream processing without direct persistence, handled by the Flow consumer
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
        cache_key_fn=task_input_hash,
        cache_expiration=timedelta(hours=1),
    )
    async def compute_js_batch(
        generator: AsyncGenerator[ESSVariableData, None],
    ) -> AsyncGenerator[ESSVariableDivergence, None]:
        """
        流式计算多个变量的 JS 散度，逐个返回每个变量的计算结果。
        Stream computation of JS divergence for multiple variables, returning
        results one by one.

        :param generator: 异步生成器，逐步提供 ESSVariableData 实例
                          Asynchronous generator providing instances of ESSVariableData
        :yield: 每个变量对应的 ESSVariableDivergence 对象
                Yields an ESSVariableDivergence object for each variable.
        """
        async for ess_data in generator:
            # 调用工具类中计算单个变量 JS 散度序列的 Task
            # Call the utility function to compute the JS divergence series for a single variable
            js_series = ESSDivergenceCalculator.compute_js_series(
                ess_data.distributions
            )
            yield ESSVariableDivergence(
                name=ess_data.name, country=ess_data.country, divergences=js_series
            )
