import json
from datetime import timedelta
from pathlib import Path
from typing import AsyncGenerator

import aiofiles
from loguru import logger

from prefect import task
from prefect.tasks import task_input_hash

from ...definition.const.prefect import LOCAL_STORAGE
from ...definition.p_model.ess_causality import ESSCausalityResult
from ...definition.p_model.ess_variable_data import ESSVariableData
from ...util.backoff import BackoffStrategy

# 创建退避策略实例
backoff = BackoffStrategy()


class ReadWriteTask:
    """
    读写任务类，包含读取 JSON 文件的任务，以及最终结果的持久化。
    Read-write task class, including tasks for reading JSON files and persisting final results.
    """

    @staticmethod
    @task(
        cache_key_fn=task_input_hash,
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
        cache_expiration=timedelta(hours=1),
        persist_result=True,
        result_storage=LOCAL_STORAGE,
    )
    async def save_causality_results_to_csv(
        result_generator: AsyncGenerator[ESSCausalityResult, None],
        output_path: str | Path,
    ):
        """
        **异步任务**：将 `ESSCausalityResult` 结果持久化到 CSV 文件。\n
        **Asynchronous task**: Persist `ESSCausalityResult` results to a CSV file.

        :param result_generator: `AsyncGenerator[ESSCausalityResult, None]`
                                 逐个提供因果分析结果的异步生成器。
                                 An asynchronous generator yielding causality analysis results.
        :param output_path: `str | Path`
                            输出 CSV 文件路径。
                            The path to the output CSV file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )  # 确保目录存在 / Ensure directory exists

        # 定义 CSV 文件的列名 / Define CSV column names
        fieldnames = ["country", "causality_from", "causality_to", "p_value"]

        try:
            async with aiofiles.open(
                output_path, mode="w", encoding="utf-8", newline=""
            ) as f:
                await f.write(
                    ",".join(fieldnames) + "\n"
                )  # 写入 CSV 头 / Write CSV header

                async for result in result_generator:
                    row = result.serialize()
                    await f.write(",".join(map(str, row.values())) + "\n")

            logger.info(
                f"因果分析结果已成功保存至 {output_path}\nCausality analysis results successfully saved to {output_path}"
            )

        except Exception as e:
            logger.error(f"写入 CSV 失败: {e}\nFailed to write CSV: {e}")
            raise e  # 让 Prefect 处理错误重试 / Let Prefect handle retries on failure

    @staticmethod
    @task(
        cache_key_fn=task_input_hash,
        retries=3,
        retry_delay_seconds=lambda attempt: backoff.fibonacci(attempt),
        cache_expiration=timedelta(hours=1),
        persist_result=True,
        result_storage=LOCAL_STORAGE,
    )
    async def load_from_json(
        file_path: str | Path,
    ) -> AsyncGenerator[ESSVariableData, None]:
        """
        **异步读取 JSON 文件**，并逐个返回 `ESSVariableData` 实例。\n
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
            data_list = json.loads(raw_data)

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
                    f"解析失败，跳过变量: {data.get('name', 'UNKNOWN')}，错误: {e}\n"
                    f"Parsing failed, skipping variable: {data.get('name', 'UNKNOWN')}, error: {e}"
                )
                continue  # 跳过错误数据 / Skip invalid data
