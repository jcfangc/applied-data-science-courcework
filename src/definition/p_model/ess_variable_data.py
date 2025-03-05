from typing import Dict

import numpy as np
from pydantic import BaseModel, Field

from ..enum.round import Round


class ESSVariableData(BaseModel):
    """Pydantic 数据模型：表示单个变量的 ESS 数据
    Pydantic data model: Represents the ESS data of a single variable.
    """

    name: str = Field(
        ..., min_length=1, description="变量分布名称 / Variable distribution name"
    )
    country: str = Field(
        ...,
        min_length=1,
        description="变量分布所属的具体国家 / The specific country to which the variable distribution belongs",
    )
    distributions: Dict[Round, np.ndarray] = Field(
        ...,
        min_items=1,
        description="该变量在不同数据集轮次的概率分布 / Probability distributions of this variable across different dataset rounds",
    )

    class Config:
        arbitrary_types_allowed = True  # 允许使用 numpy 类型 / Allow numpy types
