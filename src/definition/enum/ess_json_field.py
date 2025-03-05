from enum import Enum


class ESSJsonField(Enum):
    """枚举类，存储所有 JSON 字段
    Enum class storing all JSON fields.
    """

    VARIABLES = "variables"  # 变量列表 / List of variables
    DESCRIPTION = "description"  # 变量描述 / Variable description
    META_INFO = "meta_info"  # 元信息 / Meta information
    CODELIST = "codelist"  # 编码列表 / Codelist (value → meaning)
