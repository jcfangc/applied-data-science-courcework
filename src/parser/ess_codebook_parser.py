import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
from loguru import logger

from ..definition.const.core import DATA_DIR
from ..definition.enum.ess_json_field import ESSJsonField
from ..definition.enum.round import Round
from ..log.config import configure_logging


class ESSCodebookParser:
    def __init__(
        self, file_path: Path = DATA_DIR / Round.ESS11.value / "ESS11 codebook.html"
    ):
        """
        初始化解析器
        Initialize the parser.

        :param file_path: 本地 HTML 文件路径 / Local HTML file path
        """
        self.file_path: Path = file_path
        self.soup: Optional[BeautifulSoup] = None
        self.json_output: Dict[str, Dict] = {ESSJsonField.VARIABLES.value: {}}

        logger.info(
            f"初始化解析器，目标文件: {self.file_path}"
        )  # 初始化日志 / Initialization log

    def load_html(self) -> None:
        """加载 HTML 文件并解析为 BeautifulSoup 对象
        Load the HTML file and parse it into a BeautifulSoup object.
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.soup = BeautifulSoup(f, "html.parser")
            logger.info(
                "HTML 文件加载成功 (HTML file loaded successfully)"
            )  # HTML file loaded successfully / 日志
        except Exception as e:
            logger.error(
                f"加载 HTML 失败 (Failed to load HTML): {e}"
            )  # Failed to load HTML / 日志
            raise

    def clean_text(self, text: str) -> str:
        """
        清理文本中的多余空格和换行符，确保格式整洁
        Clean the text by removing extra spaces and line breaks to ensure tidy formatting.

        - 1. `\s+` 替换成单个空格（处理多余的空格、换行）
          Replace sequences of whitespace (`\s+`) with a single space (to handle extra spaces and line breaks)
        - 2. `strip()` 去除前后空格
          Use `strip()` to remove leading and trailing spaces.
        """
        return re.sub(r"\s+", " ", text).strip()

    def extract_description(self, h3: BeautifulSoup) -> str:
        """提取变量的描述信息，并清理多余的换行和空格
        Extract the variable's description and clean up unnecessary line breaks and spaces.
        """
        description_div = h3.find_next_sibling("div")
        description = (
            self.clean_text(description_div.get_text()) if description_div else ""
        )
        logger.debug(
            f"提取描述 (extracted description): {description}"
        )  # Log extracted description
        return description

    def extract_meta_info(self, h3: BeautifulSoup) -> List[str]:
        """提取变量的 meta 信息（多个 variable-meta-string），并清理多余的换行和空格
        Extract the variable's meta information (multiple "variable-meta-string") and clean up unnecessary line breaks and spaces.
        """
        meta_info = [
            self.clean_text(div.get_text())
            for div in h3.find_next_siblings("div", class_="variable-meta-string")
        ]
        logger.debug(
            f"提取 meta 信息 (extracted meta information): {meta_info}"
        )  # Log extracted meta information
        return meta_info

    def extract_codelist(self, h3: BeautifulSoup) -> Dict[str, str]:
        """提取变量的 codelist（值 → 含义）
        Extract the variable's codelist (value → meaning).
        """
        codelist: Dict[str, str] = {}
        data_table = h3.find_next_sibling("div", class_="data-table")
        if data_table:
            rows = data_table.find_all("tr")
            for row in rows[1:]:  # 跳过表头 / Skip the table header
                cols = row.find_all("td")
                if len(cols) == 2:
                    key = self.clean_text(cols[0].get_text(strip=True))
                    value = self.clean_text(cols[1].get_text(strip=True))
                    codelist[key] = value
        logger.debug(
            f"提取 codelist (extracted codelist): {codelist}"
        )  # Log extracted codelist
        return codelist

    def parse_variable(
        self, h3: BeautifulSoup
    ) -> Tuple[str, Dict[str, Optional[List[str]]]]:
        """解析单个变量并返回 JSON 结构
        Parse a single variable and return its JSON structure.
        """
        variable_name: str = h3.get_text(strip=True)
        logger.info(f"开始解析变量 (parsing variable): {variable_name}")

        variable_data: Dict[str, Optional[List[str]]] = {
            ESSJsonField.DESCRIPTION.value: self.extract_description(h3),
            ESSJsonField.META_INFO.value: self.extract_meta_info(h3),
            ESSJsonField.CODELIST.value: self.extract_codelist(h3),
        }

        # 移除空的 meta_info 或 codelist / Remove empty meta_info or codelist
        if not variable_data.get(ESSJsonField.META_INFO.value):
            variable_data.pop(ESSJsonField.META_INFO.value, None)
        if not variable_data.get(ESSJsonField.CODELIST.value):
            variable_data.pop(ESSJsonField.CODELIST.value, None)

        logger.info(f"解析完成 (parsing complete): {variable_name}")
        return variable_name, variable_data

    def parse(self) -> None:
        """主解析函数，遍历所有变量
        Main parsing function that iterates over all variables.
        """
        logger.info("开始解析 Codebook (starting to parse Codebook)")
        self.load_html()

        h3_elements = self.soup.find_all("h3")
        logger.info(
            f"检测到 {len(h3_elements)} 个变量 (detected {len(h3_elements)} variables)"
        )

        for h3 in h3_elements:
            try:
                var_name, var_data = self.parse_variable(h3)
                self.json_output[ESSJsonField.VARIABLES.value][var_name] = var_data
            except Exception as e:
                logger.error(
                    f"解析变量 {h3.get_text(strip=True)} 失败 (parsing variable failed): {e}"
                )

        logger.info("Codebook 解析完成 (Codebook parsing complete)")

    def save_to_json(self, output_file: str) -> None:
        """保存解析结果为 JSON
        Save the parsing result as JSON.
        """
        output_path = Path(output_file)
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.json_output, f, indent=4, ensure_ascii=False)
            logger.info(
                f"JSON 数据已保存至 {output_path} (JSON data saved to {output_path})"
            )
        except Exception as e:
            logger.error(f"保存 JSON 失败 (failed to save JSON): {e}")
            raise

    def get_json(self) -> Dict[str, Dict]:
        """返回 JSON 结构
        Return the JSON structure.
        """
        return self.json_output


if __name__ == "__main__":

    configure_logging()

    for data_set in Round:
        parser = ESSCodebookParser(
            file_path=DATA_DIR / data_set.value / f"{data_set.value} codebook.html"
        )
        parser.parse()
        parser.save_to_json(
            DATA_DIR / data_set.value / f"{data_set.value}-codebook.json"
        )
