import csv
import json
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Set,
    Tuple,
    TypeVar,
)

import aiofiles
from loguru import logger
from prefect import task

from ...definition.const.core import RESULT_DIR
from ...definition.p_model.ess_causality import ESSCausalityResult
from ...definition.p_model.ess_variable_data import ESSVariableData
from ...util.backoff import BackoffStrategy

# 创建退避策略实例
backoff = BackoffStrategy()

# 类型变量
T = TypeVar("T")


class ReadWriteTask:
    """
    读写任务类，包含读取 JSON 文件的任务，以及最终结果的持久化。
    Read-write task class, including tasks for reading JSON files and persisting final results.
    """

    @task(
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
    )
    async def load_existing_causality_results(
        output_csv: Path,
    ) -> Set[Tuple[str, str, str]]:
        """
        异步读取 CSV 结果文件，并提取已计算的因果关系数据。

        :param output_csv: CSV 结果文件的路径。
        :return: 已计算因果关系的集合，格式为 (country, var1, var2)。
        """
        existing_results: Set[Tuple[str, str, str]] = set()

        logger.info(f"正在加载已有因果关系数据: {output_csv}")

        if output_csv.exists() and output_csv.stat().st_size > 0:
            async with aiofiles.open(output_csv, mode="r", encoding="utf-8") as f:
                content = await f.readlines()  # 读取所有行
                reader = csv.reader(content)  # 用 csv.reader 解析行

                header_skipped = False
                for row in reader:
                    logger.debug(f"解析行: {row}")
                    if not header_skipped:
                        header_skipped = True  # 跳过第一行
                        continue

                    if len(row) >= 3:
                        existing_results.add((row[0], row[1], row[2]))

        logger.info(f"已加载 {len(existing_results)} 条因果关系数据")
        return existing_results

    @task(
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
    )
    async def save_causality_results_to_csv(
        results: List[ESSCausalityResult],
        output_path: str | Path = RESULT_DIR / "causality_results.csv",
    ) -> None:
        """
        将一批 ESSCausalityResult 结果追加保存到 CSV 文件。

        :param results: ESSCausalityResult 的列表。
        :param output_path: 输出 CSV 文件路径。
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["country", "causality_from", "causality_to", "p_value"]
        file_exists = output_path.exists() and output_path.stat().st_size > 0

        try:
            async with aiofiles.open(
                output_path, mode="a", encoding="utf-8", newline=""
            ) as f:
                # 如果文件不存在则写入表头
                if not file_exists:
                    await f.write(",".join(fieldnames) + "\n")
                for result in results:
                    row = result.serialize()
                    await f.write(",".join(map(str, row.values())) + "\n")
            logger.info(
                f"因果分析结果已追加保存至 {output_path}\nCausality analysis results successfully appended to {output_path}"
            )
        except Exception as e:
            logger.error(f"❌ 写入 CSV 失败: {e}\nFailed to write CSV: {e}")
            raise e

    @task(
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
    )
    async def load_from_json(
        file_path: str | Path,
    ) -> AsyncGenerator[ESSVariableData, None]:
        """
        异步读取 JSON 文件，并逐个返回 `ESSVariableData` 实例。\n
        Asynchronously reads a JSON file and yields `ESSVariableData` instances one by one.

        :param file_path: JSON 文件路径。\n
                            Path to the JSON file.
        :yield: `ESSVariableData` 实例。\n
                `ESSVariableData` instances.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"文件未找到: {file_path}\nFile not found: {file_path}"
            )

        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            logger.info(
                f"正在读取 JSON 文件: {file_path}\nReading JSON file: {file_path}"
            )
            raw_data = await f.read()
            data_list: List[Dict[str, Any]] = json.loads(raw_data)

        if not isinstance(data_list, list):
            raise ValueError(
                "JSON 文件内容应为列表格式 / JSON file content should be a list."
            )

        for data in data_list:
            try:
                yield ESSVariableData(**data)
                logger.info(
                    f"成功解析变量: {data.get('name', 'UNKNOWN')}\n"
                    f"Successfully parsed variable: {data.get('name', 'UNKNOWN')}"
                )
            except Exception as e:
                logger.error(
                    f"data: {data}\n"
                    f"解析失败，跳过变量: {data.get('name', 'UNKNOWN')}，错误: {e}\n"
                    f"Parsing failed, skipping variable: {data.get('name', 'UNKNOWN')}, error: {e}"
                )
                continue  # 跳过错误数据 / Skip invalid data

    @task(
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
    )
    async def cache_async_generator(
        generator: AsyncGenerator[T, None],
        serialize_fn: Callable[[T], Awaitable[dict]],
        cache_file: str | Path,
    ) -> Path:
        """
        将异步生成器中的数据缓存到 JSONL 文件中。\n
        Caches data from an asynchronous generator into a JSONL file.

        :param generator: 原始异步生成器。\n
                        The original asynchronous generator.
        :param serialize_fn: 异步序列化函数，将数据转换为可 JSON 序列化的字典。\n
                            An asynchronous serialization function that converts data into a JSON-serializable dictionary.
        :param cache_file: 缓存文件路径。\n
                        The path to the cache file.
        :return: 缓存文件路径。\n
                The path to the cache file.
        """
        cache_file = Path(cache_file)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.touch(exist_ok=True)

        async with aiofiles.open(cache_file, mode="a", encoding="utf-8") as f:
            async for item in generator:
                data_dict = await serialize_fn(item)
                await f.write(json.dumps(data_dict) + "\n")

        logger.info(
            f"数据已缓存到文件: {cache_file}\n"
            f"Data has been cached to file: {cache_file}"
        )
        return cache_file

    @task(
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
    )
    async def load_async_generator_from_cache(
        deserialize_fn: Callable[[dict], Awaitable[T]],
        cache_file: str | Path,
    ) -> AsyncGenerator[T, None]:
        """
        从 JSONL 缓存文件恢复异步生成器。\n
        Restores an asynchronous generator from a JSONL cache file.

        :param cache_file: 缓存文件路径。\n
                        The path to the cache file.
        :param deserialize_fn: 异步反序列化函数，将 JSON 字典恢复为原始数据类型。\n
                            An asynchronous deserialization function that converts a JSON dictionary back to the original data type.
        :yield: 异步生成器中的数据。\n
                Data from the asynchronous generator.
        """
        cache_file = Path(cache_file)

        if not cache_file.exists():
            raise FileNotFoundError(
                f"缓存文件未找到: {cache_file}\nCache file not found: {cache_file}"
            )

        async with aiofiles.open(cache_file, mode="r", encoding="utf-8") as f:
            async for line in f:
                data_dict = json.loads(line)
                item = await deserialize_fn(data_dict)
                yield item

        logger.info(
            f"已成功从缓存文件恢复数据: {cache_file}\n"
            f"Successfully restored data from cache file: {cache_file}"
        )

    @task(
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
    )
    def extract_country_name(file_path: str) -> str:
        """
        从文件路径中提取国家名。

        :param file_path: 形式为“xxx/xxx/国家名_variables.json”的文件路径
        :return: 提取的国家名
        """
        path = Path(file_path)
        filename = path.stem  # 获取文件名（不含后缀）

        if "_variables" in filename:
            return filename.replace("_variables", "")
        else:
            raise ValueError("文件名格式错误，应包含 '_variables.json'")

    @task(
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
    )
    async def extract_countries_from_folder(
        folder: Path | str,
    ) -> AsyncGenerator[str, None]:
        """
        批量获取文件夹下所有 json 文件的国家名。

        :param folder: 文件夹路径，支持 str 和 Path 类型
        :yield: 国家名
        """
        folder_path = Path(folder)

        for json_file in folder_path.glob("*.json"):  # 遍历文件夹下的所有 .json 文件
            if json_file.is_file():
                try:
                    yield ReadWriteTask.extract_country_name(json_file)
                except ValueError as e:
                    logger.warning(f"跳过文件 {json_file.name}: {e}")
