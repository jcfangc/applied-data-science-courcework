import json
import random
from typing import Any, Dict, List, Optional, Tuple

from faker import Faker
from src.definition.enum.round import Round

from ..test_core import FAKE_DATA_DIR

fake = Faker()


def _generate_codelist(num_categories: int) -> Dict[int, str]:
    """
    生成类别映射表。
    Generate a mapping of category indices to labels.

    :param num_categories: 类别数目 / Number of categories
    :return: 类别映射表 / Mapping dictionary
    """
    return {i: fake.word() for i in range(num_categories)}


def _select_continuous_rounds(
    start_pct: Optional[float] = None, end_pct: Optional[float] = None
) -> List[Round]:
    """
    根据可选的起始和结束百分比或者随机选择，返回连续的轮次列表。
    Select a continuous block of rounds from the Round enum based on optional start and end percentages.

    :param start_pct: 起始百分比（0 到 1），用于计算起始下标 / Start percentage (0 to 1)
    :param end_pct: 结束百分比（0 到 1），用于计算结束下标 / End percentage (0 to 1)
    :return: 连续的轮次列表 / List of consecutive rounds
    """
    round_list = list(Round)
    n_rounds = len(round_list)

    if start_pct is not None and end_pct is not None:
        # 计算下标，四舍五入
        # Calculate indices, rounding off
        start_index = round(start_pct * n_rounds)
        end_index = round(end_pct * n_rounds)
        start_index = max(0, start_index)
        end_index = min(n_rounds, end_index)
        # 确保至少有2个连续轮次
        # Ensure at least 2 consecutive rounds
        if end_index - start_index < 2:
            end_index = min(n_rounds, start_index + 2)
        return round_list[start_index:end_index]
    else:
        # 随机选择连续轮次
        start_index = random.randint(0, n_rounds - 5)  # 确保至少有 5 个可选
        end_index = start_index + random.randint(2, 5)  # 确保选择 2~5 个连续的轮次
        return round_list[start_index:end_index]


def generate_normalized_distribution(num_categories: int) -> List[float]:
    """
    生成归一化的概率分布，保留小数点后 4 位。
    Generate a normalized probability distribution with 4 decimal places.

    :param num_categories: 类别数量 / Number of categories
    :return: 概率分布列表 / List of probabilities
    """
    prob_distribution = [round(random.uniform(0, 1), 4) for _ in range(num_categories)]
    total = sum(prob_distribution)
    return [round(p / total, 4) for p in prob_distribution]


def generate_fake_ess_variable_data(
    start_pct: Optional[float] = None, end_pct: Optional[float] = None
) -> Dict[str, Any]:
    """
    生成符合 `ESSVariableData` 结构的假数据 JSON，并允许传入可选的起始和结束百分比来控制连续轮次下标。
    Generate fake JSON data compatible with `ESSVariableData`, with optional start and end percentages
    to control the indices for selecting a continuous block of rounds.

    :param start_pct: 起始百分比（0 到 1），用于计算起始下标 / Start percentage (0 to 1)
    :param end_pct: 结束百分比（0 到 1），用于计算结束下标 / End percentage (0 to 1)
    :return: 假数据字典 / Fake data dictionary
    """
    name = fake.word()
    country = "UK"

    # 生成 codelist (类别索引 -> 说明) / Generate codelist (category index -> label)
    num_categories = random.randint(2, 6)  # 2~6 个类别 / 2~6 categories
    codelist = _generate_codelist(num_categories)

    # 根据百分比或随机选择连续轮次
    selected_rounds = _select_continuous_rounds(start_pct, end_pct)

    # 为每个选定的轮次生成归一化的概率分布 / Generate normalized probability distribution for each selected round
    distributions = {}
    for round_enum in selected_rounds:
        normalized_distribution = generate_normalized_distribution(num_categories)
        distributions[round_enum.value] = normalized_distribution

    return {
        "name": name,
        "country": country,
        "codelist": codelist,
        "distributions": distributions,
    }


def generate_fake_ess_variable_data_list(
    count: int = 10,
    round_range_pct: Optional[Tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    """
    生成包含多个 `ESSVariableData` 兼容的假数据的列表，并允许所有数据共享相同的轮次。
    Generate a list of fake JSON data compatible with `ESSVariableData`, allowing consistent rounds across all records.

    :param count: 生成的数据条数 (默认 10)
                  Number of records to generate (default 10).
    :param round_range_pct: 可选的 (start_pct, end_pct) 范围，用百分比控制候选轮次在枚举中的起始结束位置，
                            范围应在 [0, 1] 内，且 start_pct < end_pct。
                            Optional tuple (start_pct, end_pct) controlling the candidate rounds selection as a percentage
                            of the total available rounds. Values should be in [0, 1] and start_pct < end_pct.
    :return: 假数据列表 / List of fake data dictionaries.
    """
    return [
        generate_fake_ess_variable_data(
            start_pct=round_range_pct[0], end_pct=round_range_pct[1]
        )
        for _ in range(count)
    ]


if __name__ == "__main__":
    # 生成一个测试用 JSON
    fake_data_json = generate_fake_ess_variable_data_list(
        count=20, round_range_pct=(0, 1)
    )

    # 保存到文件
    with open(FAKE_DATA_DIR / "fake_ess_variable_data.json", "w") as f:
        json.dump(fake_data_json, f, indent=2)
