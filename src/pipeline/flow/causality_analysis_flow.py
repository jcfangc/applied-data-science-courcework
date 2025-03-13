from pathlib import Path

from prefect import flow

from ...definition.const.core import DIVERGENCE_CACHE_DIR, RESULT_DIR
from ..task.adjacency_matrix import AdjacencyMatrixTask
from ..task.granger_causality import ESSCausalityCalculatorTask
from ..task.js_divergence import ESSDivergenceCalculatorTask
from ..task.read_write import ReadWriteTask


@flow
async def causality_analysis_flow(
    json_dir: Path,
    js_divergence_dir: Path = DIVERGENCE_CACHE_DIR,
    gg_causality_dir: Path = RESULT_DIR,
    maxlag: int = 2,
) -> None:
    async for country in ReadWriteTask.extract_countries_from_folder(folder=json_dir):
        # Step 1: 加载 JSON 数据
        # Step 1: Load JSON data
        json_file = json_dir / f"{country}_variables.json"
        ess_data_generator = ReadWriteTask.load_from_json(file_path=json_file)

        # Step 2: 计算 JS 散度批次
        # Step 2: Compute JS divergence batch
        divergence_cache_file = js_divergence_dir / f"{country}_divergences.jsonl"
        js_divergences_list = await ESSDivergenceCalculatorTask.compute_js_batch(
            generator=ess_data_generator, cache_file=divergence_cache_file
        )

        # Step 3: 构建因果邻接表
        # Step 3: Build causality adjacency matrices
        adjacency_matrices = (
            await AdjacencyMatrixTask.build_causality_adjacency_matrices(
                divergences_list=js_divergences_list
            )
        )

        # Step 4: 提取邻接表中的计算条目
        # Step 4: Extract computation entries from adjacency matrices
        adjacency_entries_generator = AdjacencyMatrixTask.extract_adjacency_entries(
            adjacency_matrices=adjacency_matrices
        )

        # Step 5: 根据邻接表计算因果关系
        # Step 5: Compute causality relationships based on adjacency matrices
        result_csv = gg_causality_dir / f"{country}_causality.csv"
        # 生成动态文件名（在原文件名后追加 "_lag{maxlag}.csv"）
        # Generate dynamic file name (append "_lag{maxlag}.csv" to the original file name)
        result_csv = result_csv.with_stem(result_csv.stem + f"_lag{maxlag}")
        await ESSCausalityCalculatorTask.compute_causal_relationship_from_adjacency(
            adjacency_matrices=adjacency_matrices,
            adjacency_entries_generator=adjacency_entries_generator,
            divergences_list=js_divergences_list,
            output_csv=result_csv,
            maxlag=maxlag,
        )
