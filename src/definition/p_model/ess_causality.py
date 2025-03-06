from pydantic import BaseModel, Field, model_serializer


class ESSCausalityResult(BaseModel):
    """
    表示 Granger 因果分析的结果，包含国家、因果关系的起点变量、终点变量和 p 值。
    Represents the result of a Granger causality analysis, including the country,
    the causality source variable, the target variable, and the p-value.
    """

    country: str = Field(
        ..., min_length=1, description="所属国家 / Country of the analysis"
    )
    causality_from: str = Field(
        ..., min_length=1, description="因果关系起点变量 / Causality source variable"
    )
    causality_to: str = Field(
        ..., min_length=1, description="因果关系终点变量 / Causality target variable"
    )
    p_value: float = Field(
        ...,
        ge=0,
        le=1,
        description="Granger 因果分析的 p 值 / P-value from Granger causality analysis",
    )

    @model_serializer
    def serialize(self) -> dict:
        """
        序列化模型为字典格式 / Serialize the model into a dictionary format.
        """
        return {
            "country": self.country,
            "causality_from": self.causality_from,
            "causality_to": self.causality_to,
            "p_value": self.p_value,
        }
