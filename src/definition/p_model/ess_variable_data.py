from typing import Dict, Optional

import polars as pl
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
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
    codelist: Optional[Dict[int, str]] = Field(
        None,
        description="变量的类别映射表 (整数索引 -> 选项说明)，可为空 / Variable category mapping (int -> label), can be empty",
    )
    distributions: Dict[Round, pl.Series] = Field(
        ...,
        min_length=1,
        description="该变量在不同数据集轮次的概率分布 / Probability distributions of this variable across different dataset rounds",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_serializer
    def serialize(self):
        """
        序列化 `ESSVariableData`，确保 `pl.Series` 被正确转换为 JSON 兼容格式。
        Serialize `ESSVariableData`, ensuring `pl.Series` is properly converted to JSON.
        """
        return {
            "name": self.name,
            "country": self.country,
            "codelist": self.codelist or {},  # 确保 codelist 是一个字典，即使为空
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
        cls, value: Dict[Round, pl.Series], info: ValidationInfo
    ) -> Dict[Round, pl.Series]:
        """
        确保所有 `distributions` 的 `pl.Series` 长度一致。
        1. 如果 `codelist` 存在，`distributions` 长度必须匹配 `codelist`。
        2. 如果 `codelist` 为空，确保 `distributions` 内所有 `pl.Series` 长度一致。
        Ensure all `pl.Series` in `distributions` have consistent length.
        """
        codelist = info.data.get("codelist", None)

        # 1️⃣ 获取第一个 `pl.Series` 作为标准长度
        lengths = {len(series) for series in value.values()}

        if len(lengths) > 1:
            raise ValueError(
                f"所有 distributions 的 `pl.Series` 长度必须一致，当前长度集合: {lengths} / "
                f"All distributions `pl.Series` must have the same length, but found: {lengths}"
            )

        # 2️⃣ 如果 `codelist` 存在，检查 `pl.Series` 长度是否匹配 `codelist`
        if codelist is not None:
            codelist_length = len(codelist)
            expected_length = next(iter(lengths))  # 获取第一个 `pl.Series` 长度

            if codelist_length != expected_length:
                raise ValueError(
                    f"codelist 长度 ({codelist_length}) 与 `distributions` 长度 ({expected_length}) 不匹配 / "
                    f"codelist length ({codelist_length}) does not match `distributions` length ({expected_length})"
                )

        return value
