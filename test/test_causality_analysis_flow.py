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
    fake_data_dir = FAKE_DATA_DIR
    fake_js_cache_dir = FAKE_CACHE_DIR
    causality_csv_dir = FAKE_RESULT_DIR

    # 调用 flow 执行任务
    await causality_analysis_flow(
        json_dir=fake_data_dir,
        js_divergence_dir=fake_js_cache_dir,
        gg_causality_dir=causality_csv_dir,
        maxlag=maxlag,
    )
