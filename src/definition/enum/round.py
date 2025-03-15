import re
from enum import Enum
from typing import List


class Round(Enum):
    """定义 ESS 数据集枚举，包括插值轮次
    Define the ESS dataset enumeration, including interpolated rounds.
    """

    ESS1 = "ESS1e06_7"
    ESS1_5 = "ESS1.5"
    ESS2 = "ESS2e03_6"
    ESS2_5 = "ESS2.5"
    ESS3 = "ESS3e03_7"
    ESS3_5 = "ESS3.5"
    ESS4 = "ESS4e04_6"
    ESS4_5 = "ESS4.5"
    ESS5 = "ESS5e03_5"
    ESS5_5 = "ESS5.5"
    ESS6 = "ESS6e02_6"
    ESS6_5 = "ESS6.5"
    ESS7 = "ESS7e02_3"
    ESS7_5 = "ESS7.5"
    ESS8 = "ESS8e02_3"
    ESS8_5 = "ESS8.5"
    ESS9 = "ESS9e03_2"
    ESS9_5 = "ESS9.5"
    ESS10 = "ESS10"
    ESS10_5 = "ESS10.5"
    ESS11 = "ESS11"

    @classmethod
    def extract_digits(cls, value: str) -> float:
        """
        提取字符串中的数字部分，允许浮点数，用于匹配 `Round`。
        在此基础上，先将下划线替换为小数点，以支持诸如 `ESS1_5` -> `ESS1.5`.
        Extract the numeric part from a string, allowing floating-point values, for matching `Round`.
        First replace underscores with dots to support e.g. `ESS1_5` -> `ESS1.5`.
        """
        # 将下划线视为小数点
        value = value.replace("_", ".")
        match = re.search(r"(\d+(\.\d+)?)", value)  # 匹配整数或小数
        return float(match.group(1)) if match else None

    @classmethod
    def from_str(cls, value: str) -> "Round":
        """
        通过字符串获取对应的枚举值，允许匹配插值轮次（如 `ESS5.5`）。
        Get the corresponding enum value from a string, allowing interpolated rounds (e.g., `ESS5.5`).
        """
        input_digits = cls.extract_digits(value)
        if input_digits is None:
            raise ValueError(
                f"无法从输入 `{value}` 提取数字部分 (Cannot extract numeric part from `{value}`)"
            )

        for dataset in cls:
            round_digits = cls.extract_digits(dataset.name)
            if round_digits == input_digits:
                return dataset

        raise ValueError(f"未知的数据集标识符 (Unknown dataset identifier): {value}")

    @classmethod
    def get_interpolated_round(cls, r: "Round") -> "Round":
        """
        将整数轮次（例如 ESS1）映射到插值轮次（例如 ESS1_5）。
        Mapping integer rounds (e.g., ESS1) to interpolated rounds (e.g., ESS1_5).
        要求 Round 枚举中已经定义了对应的插值轮次，否则抛出 KeyError。
        Raises KeyError if the corresponding interpolated round is not defined in the Round enum.

        :param r: Round 对象，比如 Round.ESS1
                Round object, e.g., Round.ESS1
        :return: 对应的插值 Round，例如 Round.ESS1_5
                Corresponding interpolated Round, e.g., Round.ESS1_5
        :raises ValueError: 如果提取到的不是整数或无法提取数字
                            If the extracted value is not an integer or cannot extract a number
        :raises KeyError: 如果对应的插值枚举未定义
                            If the corresponding interpolated enum is not defined
        """
        digits = cls.extract_digits(r.name)  # 例如 "ESS1" 提取到 float(1.0)
        if digits is None:
            # 没有找到数字
            raise ValueError(f"无法从 {r.name} 中提取数字，无法进行插值。")

        # 检查是否为整数
        if not digits.is_integer():
            raise ValueError(
                f"枚举 {r.name} 中的数字不是整数 (解析为 {digits})，无法映射到插值轮次。"
            )

        # 将整数部分组合成插值轮次，如 ESS1 -> ESS1_5
        round_str = f"ESS{int(digits)}_5"
        return cls[round_str]

    @classmethod
    def get_official_rounds(cls) -> List["Round"]:
        """
        获取正式轮次（不包含插值轮次）。
        Get official rounds (excluding interpolated rounds).

        :return: 仅包含正式轮次的 `Round` 枚举列表。
        """
        return [r for r in cls if "_5" not in r.name]

    @classmethod
    def get_interpolated_rounds(cls) -> List["Round"]:
        """
        获取插值轮次（如 ESS1_5, ESS2_5）。
        Get interpolated rounds (e.g., ESS1_5, ESS2_5).

        :return: 仅包含插值轮次的 `Round` 枚举列表。
        """
        return [r for r in cls if "_5" in r.name]

    def __str__(self) -> str:
        """保证 Round 输出时仍然格式化为 ESSx.x 形式"""
        return self.name.replace("_", ".")
