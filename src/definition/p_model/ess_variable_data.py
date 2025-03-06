from typing import Any, Dict

import polars as pl
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_serializer,
)

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
    codelist: Dict[int, str] = Field(
        ...,
        description="变量的类别映射表 (整数索引 -> 选项说明) / Variable category mapping (int -> label)",
    )
    distributions: Dict[Round, pl.Series] = Field(
        ...,
        min_items=1,
        description="该变量在不同数据集轮次的概率分布 / Probability distributions of this variable across different dataset rounds",
    )

    class Config:
        arbitrary_types_allowed = True  # 允许使用 Polars 类型 / Allow Polars types

    @model_serializer
    def serialize(self):
        """
        序列化 `ESSVariableData`，确保 `pl.Series` 被正确转换为 JSON 兼容格式。
        Serialize `ESSVariableData`, ensuring `pl.Series` is properly converted to JSON.
        """
        return {
            "name": self.name,
            "country": self.country,
            "codelist": self.codelist,
            "distributions": {
                str(round_): dist.to_list()
                for round_, dist in self.distributions.items()
            },
        }

    @field_validator("distributions", mode="before")
    @classmethod
    def deserialize_distributions(
        cls, value: Dict[str, list]
    ) -> Dict[Round, pl.Series]:
        """
        反序列化 JSON 格式的 `distributions`，将其转换为 `Dict[Round, pl.Series]`
        Deserialize `distributions` from JSON format to `Dict[Round, pl.Series]`
        """
        if not isinstance(value, dict):
            raise ValueError("distributions 必须是一个字典")

        parsed_distributions = {}
        for round_key, data in value.items():
            try:
                round_enum = Round.from_str(round_key)
            except ValueError as e:
                raise ValueError(
                    f"无效的轮次标识: {round_key} / Invalid round identifier: {round_key}"
                ) from e

            if not isinstance(data, list):
                raise ValueError(
                    f"轮次 {round_key} 的数据格式应为列表 / "
                    f"The data format for round {round_key} must be a list."
                )

            parsed_distributions[round_enum] = pl.Series(data)

        return parsed_distributions

    @field_validator("distributions", mode="after")
    @classmethod
    def validate_distribution_length(
        cls, value: Dict[Round, pl.Series], values: Dict[str, Any]
    ) -> Dict[Round, pl.Series]:
        """
        确保所有 distributions 的 pl.Series 长度和 codelist 的长度一致
        Ensure all pl.Series in distributions have the same length as codelist.
        """
        codelist_length: int = len(values.get("codelist", {}))

        for round_key, series in value.items():
            if not isinstance(series, pl.Series):
                raise ValueError(
                    f"轮次 {round_key} 的分布必须是 polars.Series 类型 / "
                    f"Distribution for round {round_key} must be of type polars.Series"
                )
            if len(series) != codelist_length:
                raise ValueError(
                    f"轮次 {round_key} 的分布长度 ({len(series)}) 与 codelist 长度 ({codelist_length}) 不匹配 / "
                    f"Length of distribution for round {round_key} ({len(series)}) does not match codelist length ({codelist_length})"
                )

        return value
