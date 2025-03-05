from enum import Enum


class Round(Enum):
    """定义 ESS 数据集枚举
    Define the ESS dataset enumeration.
    """

    ESS1 = "ESS1e06_7"
    ESS2 = "ESS2e03_6"
    ESS3 = "ESS3e03_7"
    ESS4 = "ESS4e04_6"
    ESS5 = "ESS5e03_5"
    ESS6 = "ESS6e02_6"
    ESS7 = "ESS7e02_3"
    ESS8 = "ESS8e02_3"
    ESS9 = "ESS9e03_2"
    ESS10 = "ESS10"
    ESS11 = "ESS11"

    @classmethod
    def from_str(cls, value: str) -> "Round":
        """通过字符串获取对应的枚举值
        Get the corresponding enum value from a string.
        """
        for dataset in cls:
            if dataset.value == value:
                return dataset
        raise ValueError(f"未知的数据集标识符 (Unknown dataset identifier): {value}")
