from pathlib import Path

from prefect import flow

from ...definition.const.core import CAUSALITY_DIR, GENERATOR_CACHE_DIR
from ...definition.p_model.ess_divergence import ESSVariableDivergences
from ..task.adjacency_matrix import AdjacencyMatrixTask
from ..task.granger_causality import ESSCausalityCalculatorTask
from ..task.js_divergence import ESSDivergenceCalculatorTask
from ..task.read_write import ReadWriteTask


@flow
async def causality_analysis_flow(
    json_file: Path,
    js_cache_file: Path = GENERATOR_CACHE_DIR / "js_divergence.jsonl",
    causality_results_csv: Path = CAUSALITY_DIR / "granger_causality_results.csv",
    maxlag: int = 2,
) -> None:
    # Step 1: 加载 JSON 数据
    # Step 1: Load JSON data
    ess_data_generator = ReadWriteTask.load_from_json(file_path=json_file)

    # Step 2: 计算 JS 散度批次
    # Step 2: Compute JS divergence batch
    js_divergence_generator = ESSDivergenceCalculatorTask.compute_js_batch(
        generator=ess_data_generator,
    )

    # Step 3: 缓存 JS 散度结果到文件
    # Step 3: Cache JS divergence results to file
    await ReadWriteTask.cache_async_generator(
        generator=js_divergence_generator,
        serialize_fn=lambda data: data.serialize(),
        cache_file=js_cache_file,
    )

    # Step 4: 从缓存加载 JS 散度数据
    # Step 4: Load JS divergence data from cache
    js_divergence_cached_generator = ReadWriteTask.load_async_generator_from_cache(
        deserialize_fn=lambda data: ESSVariableDivergences.model_validate(data),
        cache_file=js_cache_file,
    )

    # Step 5: 构建因果邻接表
    # Step 5: Build causality adjacency matrices
    adjacency_matrices = await AdjacencyMatrixTask.build_causality_adjacency_matrices(
        generator=js_divergence_cached_generator
    )

    # Step 6: 提取邻接表中的计算条目
    # Step 6: Extract computation entries from adjacency matrices
    adjacency_entries_generator = AdjacencyMatrixTask.extract_adjacency_entries(
        adjacency_matrices=adjacency_matrices
    )

    # Step 7: 根据邻接表计算因果关系
    # Step 7: Compute causality relationships based on adjacency matrices
    causality_result_generator = (
        ESSCausalityCalculatorTask.compute_causal_relationship_from_adjacency(
            adjacency_matrices=adjacency_matrices,
            adjacency_entries_generator=adjacency_entries_generator,
            js_divergence_gen_cache_file=js_cache_file,
            deserialize_fn=lambda data: ESSVariableDivergences.model_validate(data),
            maxlag=maxlag,
        )
    )

    # 生成动态文件名（在原文件名后追加 "_lag{maxlag}.csv"）
    # Generate dynamic file name (append "_lag{maxlag}.csv" to the original file name)
    causality_results_csv = causality_results_csv.with_stem(
        causality_results_csv.stem + f"_lag{maxlag}"
    )

    # Step 8: 保存因果分析结果到 CSV 文件
    # Step 8: Save causality analysis results to CSV file
    await ReadWriteTask.save_causality_results_to_csv(
        result_generator=causality_result_generator,
        output_path=causality_results_csv,
    )
