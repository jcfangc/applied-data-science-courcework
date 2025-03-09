import random
from typing import Callable

from ..definition.enum.backoff_mode import BackoffMode


class BackoffStrategy:
    """
    退避策略类，提供多种重试退避算法（Fibonacci、指数退避等）。
    Backoff strategy class, providing multiple retry backoff algorithms (Fibonacci, Exponential, etc.).
    """

    def __init__(self, jitter_range: float = 0.5):
        """
        初始化退避策略。
        Initialize the backoff strategy.

        :param jitter_range: 随机抖动范围（默认 ±0.5 秒）
                             Range for random jitter (default ±0.5 seconds)
        """
        self.jitter_range = jitter_range
        self._fibonacci_cache = {}  # 用于缓存 Fibonacci 基础值 / Cache for Fibonacci base values

    def _add_jitter(self, value: float) -> int:
        """
        给定一个值 `value`，在其基础上加入对称的随机抖动（范围 ± jitter_range），
        并返回四舍五入后的整数。

        Add symmetric random jitter within ± jitter_range to the given value,
        and return the result as an integer.

        :param value: 原始退避时间 / Original backoff time.
        :return: 加入随机抖动后并四舍五入的整数值 / Integer value with added random jitter.
        """
        jittered = value + random.uniform(-self.jitter_range, self.jitter_range)
        return int(round(jittered))

    def fibonacci(self, attempt: int) -> int:
        """
        计算 Fibonacci 退避时间，带对称随机抖动，并返回整数。

        Compute the Fibonacci backoff time with symmetric random jitter, returning an integer.

        :param attempt: 当前重试次数 / Current retry attempt.
        :return: 退避时间（秒） / Backoff time in seconds (integer).
        """
        if attempt < 0:
            raise ValueError("Attempt must be >= 0")

        if attempt in self._fibonacci_cache:
            base_value = self._fibonacci_cache[attempt]
        else:
            if attempt == 0 or attempt == 1:
                base_value = 1
            else:
                a, b = 1, 1
                for _ in range(2, attempt + 1):
                    a, b = b, a + b
                base_value = b
            self._fibonacci_cache[attempt] = base_value

        return self._add_jitter(base_value)

    def exponential(self, attempt: int, base: float = 1.0, factor: float = 2.0) -> int:
        """
        计算指数退避时间，带对称随机抖动，并返回整数。

        Compute the exponential backoff time with symmetric random jitter, returning an integer.

        :param attempt: 当前重试次数 / Current retry attempt.
        :param base: 基础延迟时间 / Base delay time.
        :param factor: 指数增长因子 / Exponential growth factor.
        :return: 退避时间（秒） / Backoff time in seconds (integer).
        """
        if attempt < 0:
            raise ValueError("Attempt must be >= 0")
        return self._add_jitter(base * (factor**attempt))

    def get_backoff_function(
        self, mode: BackoffMode = BackoffMode.FIBONACCI
    ) -> Callable[[int], int]:
        """
        根据 `mode` 返回对应的退避计算函数，返回值为整数。

        Return the corresponding backoff calculation function based on `mode`, returning an integer.

        :param mode: BackoffMode 枚举值，可选 BackoffMode.FIBONACCI 或 BackoffMode.EXPONENTIAL.
        :return: 计算退避时间的函数 / Function to calculate backoff time.
        """
        if mode == BackoffMode.FIBONACCI:
            return self.fibonacci
        elif mode == BackoffMode.EXPONENTIAL:
            return lambda attempt: self.exponential(attempt)
        else:
            raise ValueError("不支持的退让策略 / Unsupported backoff strategy")
