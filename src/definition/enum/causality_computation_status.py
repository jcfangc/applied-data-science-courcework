from enum import Enum


class CausalityComputationStatus(Enum):
    """因果计算状态枚举
    Causality computation status enumeration.
    """

    FAILED = "failed"  # 计算失败 / Computation failed
    IN_PROGRESS = "in_progress"  # 计算中 / Computation in progress
    SUCCESS = "success"  # 计算成功 / Computation succeeded
