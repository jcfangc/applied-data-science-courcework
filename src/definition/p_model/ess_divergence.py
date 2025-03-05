from typing import List

from pydantic import BaseModel, Field

from ..enum.round import Round


class ESSSingleDivergence(BaseModel):
    """表示两个相邻轮次之间的 JS 散度
    Represents the JS divergence between two adjacent rounds.
    """

    round_from: Round = Field(..., description="起始轮次 / Starting round")
    round_to: Round = Field(..., description="目标轮次 / Target round")
    js_divergence: float = Field(
        ...,
        ge=0,
        le=0.7,
        description="JS 散度值，取值范围为 [0, log(2)] / JS divergence value, range [0, log(2)]",
    )


class ESSVariableDivergence(BaseModel):
    """
    表示单个变量的 JS 散度结果，包含变量名称和对应的散度列表。
    Represents the JS divergence result of a single variable,
    including the variable name and corresponding divergence list.
    """

    name: str = Field(
        ...,
        min_length=1,
        description="变量名称，不能为空 / Variable name, cannot be empty",
    )
    country: str = Field(
        ...,
        min_length=1,
        description="变量分布所属的具体国家 / The specific country to which the variable distribution belongs",
    )
    divergences: List[ESSSingleDivergence] = Field(
        ...,
        min_items=1,
        description="该变量在不同轮次间的 JS 散度列表，至少包含一个散度值 / List of JS divergences between different rounds for this variable, must contain at least one divergence value",
    )
