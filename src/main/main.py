import asyncio

from ..definition.const.core import (
    CAUSALITY_RESULT_DIR,
    DIVERGENCE_CACHE_DIR,
    ROOT_DIR,
)
from ..log.config import configure_logging
from ..pipeline.flow.causality_analysis_flow import causality_analysis_flow


async def main():
    print("Running main function...")
    configure_logging()

    MAXLAG = 2
    SOURCE_DATA_DIR = ROOT_DIR / "Data_cleaning" / "countries json file"

    # 调用 flow 执行任务
    await causality_analysis_flow(
        json_dir=SOURCE_DATA_DIR,
        js_divergence_dir=DIVERGENCE_CACHE_DIR,
        gg_causality_dir=CAUSALITY_RESULT_DIR,
        maxlag=MAXLAG,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print("Error during execution:", e)
        raise e
