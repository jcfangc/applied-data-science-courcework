import pytest
from src.log.config import configure_logging
from src.pipeline.flow.causality_analysis_flow import causality_analysis_flow
from test.test_core import FAKE_CACHE_DIR, FAKE_DATA_DIR, FAKE_RESULT_DIR


@pytest.mark.asyncio
async def test_causality_analysis_flow():
    """
    使用准备好的伪数据进行测试：
    - 读取伪数据文件
    - 调用 flow 运行因果分析任务
    - 检查输出的 CSV 文件是否存在且不为空
    """

    configure_logging()

    maxlag = 2

    # 定义文件路径
    fake_data_file = FAKE_DATA_DIR / "fake_ess_variable_data.json"
    js_cache_file = FAKE_CACHE_DIR / "js_divergence_test.jsonl"
    causality_csv_file = FAKE_RESULT_DIR / "granger_causality_results_test.csv"

    final_causality_csv_file = causality_csv_file.with_stem(
        causality_csv_file.stem + f"_lag{maxlag}"
    )

    # 如果目标文件已存在，先删除（避免测试受上次结果影响）
    if final_causality_csv_file.exists():
        final_causality_csv_file.unlink()
    if js_cache_file.exists():
        js_cache_file.unlink()

    # 调用 flow 执行任务
    await causality_analysis_flow(
        json_file=fake_data_file,
        js_cache_file=js_cache_file,
        causality_results_csv=causality_csv_file,
        maxlag=maxlag,
    )

    # 验证输出 CSV 文件是否存在
    assert final_causality_csv_file.exists(), "因果分析结果 CSV 文件未生成！"

    # 验证 CSV 文件内容不为空
    with open(final_causality_csv_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
    assert content, "因果分析结果 CSV 文件内容为空！"

    print("测试通过：因果分析 flow 正常运行且输出文件不为空。")


# if __name__ == "__main__":
#     test_causality_analysis_flow()
