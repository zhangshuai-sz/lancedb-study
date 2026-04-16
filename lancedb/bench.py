"""Benchmark LanceDB using direct async client calls (no FastAPI)."""

import argparse
import asyncio
import gc
import os
import random
import sys
import warnings
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from time import perf_counter

# 抑制 Rust 端的 Lance 警告
os.environ.setdefault("RUST_LOG", "error")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="lancedb")

import lancedb

try:
    from .config import Settings
except ImportError:
    from config import Settings

# 日志输出文件路径
LOG_FILE = Path(__file__).resolve().parents[1] / "bench_output.log"

QUERY_FILES = {
    "fts": "keyword_terms.txt",
    "vector": "keyword_terms.txt",
    "hybrid": "keyword_terms.txt",
}
SEARCH_TYPES = ("fts", "vector", "hybrid")
NUM_QUERIES = 1000
NUM_TRIALS = 3
DEFAULT_SEED = 37
EMBEDDING_DIM = 1024
RESULT_COLUMNS = ["id", "title", "text", "url", "wiki_id", "views", "paragraph_id", "langs"]
FTS_RESULT_COLUMNS = [*RESULT_COLUMNS, "_score"]
VECTOR_RESULT_COLUMNS = [*RESULT_COLUMNS, "_distance"]
HYBRID_RESULT_COLUMNS = [*RESULT_COLUMNS, "_score"]


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def get_query_terms(search_type: str) -> list[str]:
    query_dir = Path(__file__).resolve().parents[1] / "bench_queries"
    query_terms_file = query_dir / QUERY_FILES[search_type]
    with open(query_terms_file, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f.readlines() if line.strip()]
    if not queries:
        raise ValueError(f"No benchmark queries found in {query_terms_file}")
    return queries


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (p / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


async def fetch_random_vectors(
    table: lancedb.table.AsyncTable,
    n: int,
) -> list[list[float]]:
    """从 LanceDB 表中随机抽取 n 条文档的向量，用作向量检索的查询向量。"""
    import pyarrow.compute as pc

    # 使用 LanceDB 的 sample 或直接取前 N 条的向量
    result = await table.query().select(["vector"]).limit(n * 2).to_arrow()
    vectors = result.column("vector").to_pylist()
    if not vectors:
        raise RuntimeError("无法从 LanceDB 中获取向量数据，请确认数据已导入")
    # 随机打乱
    random.shuffle(vectors)
    return vectors[:n]


async def run_lancedb_fts_query(table: lancedb.table.AsyncTable, query_text: str) -> bool:
    query = await table.search(
        query_text,
        query_type="fts",
        fts_columns=["text"],
    )
    result_table = await query.select(FTS_RESULT_COLUMNS).limit(10).to_arrow()
    return result_table.num_rows > 0


async def run_lancedb_vector_query(
    table: lancedb.table.AsyncTable,
    query_vector: list[float],
) -> bool:
    query = await table.search(
        query_vector,
        vector_column_name="vector",
        query_type="vector",
    )
    result_table = await (
        query.distance_type("cosine")
        .nprobes(128)
        .select(VECTOR_RESULT_COLUMNS)
        .limit(10)
        .to_arrow()
    )
    return result_table.num_rows > 0


async def run_lancedb_hybrid_query(
    table: lancedb.table.AsyncTable,
    query_text: str,
    query_vector: list[float],
) -> bool:
    builder = (
        table.query()
        .nearest_to(query_vector)
        .column("vector")
        .distance_type("cosine")
        .nprobes(32)
        .nearest_to_text(query_text, columns=["text"])
        .limit(10)
    )
    result_table = await builder.to_arrow()
    return result_table.num_rows > 0


async def run_single_trial(
    sampled_queries: list[str],
    warmup_queries: list[str],
    max_concurrency: int,
    query_runner,
    batch_size: int | None = None,
    progress_label: str = "",
) -> dict[str, float]:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def timed_query(query_text: str) -> tuple[bool, float]:
        async with semaphore:
            start = perf_counter()
            ok = await query_runner(query_text)
            latency_ms = (perf_counter() - start) * 1000
            return ok, latency_ms

    if warmup_queries:
        await asyncio.gather(*(timed_query(query) for query in warmup_queries))

    # 分批执行查询，避免一次性提交过多任务导致内存暴涨
    batch_size = batch_size or len(sampled_queries)  # 默认不分批
    total_queries = len(sampled_queries)
    num_batches = (total_queries + batch_size - 1) // batch_size
    start_total = perf_counter()
    results = []
    for i in range(0, total_queries, batch_size):
        batch = sampled_queries[i : i + batch_size]
        batch_results = await asyncio.gather(*(timed_query(query) for query in batch))
        results.extend(batch_results)
        # 分批模式下输出进度日志（仅当有多个批次时）
        if num_batches > 1 and progress_label:
            done = min(i + batch_size, total_queries)
            elapsed_so_far = perf_counter() - start_total
            batch_latencies = [lat for _, lat in batch_results]
            avg_lat = sum(batch_latencies) / len(batch_latencies) if batch_latencies else 0
            current_qps = done / elapsed_so_far if elapsed_so_far > 0 else 0
            print(
                f"  [{progress_label}] {done}/{total_queries} 完成 "
                f"| 已耗时 {elapsed_so_far:.1f}s | 当前QPS {current_qps:.1f} "
                f"| 本批avg {avg_lat:.1f}ms",
                flush=True,
            )
    elapsed_total = perf_counter() - start_total

    successes = sum(1 for ok, _ in results if ok)
    latencies_ms = [latency for _, latency in results]

    return {
        "success": float(successes),
        "elapsed_s": elapsed_total,
        "qps": NUM_QUERIES / elapsed_total if elapsed_total > 0 else float("inf"),
        "p50_ms": percentile(latencies_ms, 50),
        "p95_ms": percentile(latencies_ms, 95),
        "p99_ms": percentile(latencies_ms, 99),
    }


def average_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    keys = metrics_list[0].keys()
    return {
        key: sum(metrics[key] for metrics in metrics_list) / len(metrics_list)
        for key in keys
    }


async def run_benchmark(args: argparse.Namespace) -> None:
    settings = get_settings()

    db_uri = str(Path(__file__).resolve().parent / settings.lancedb_dir)

    # 先连接一次，抽取向量数据（后续会关闭重连）
    db = await lancedb.connect_async(db_uri)
    table = await db.open_table("wikipedia_dedup")

    print(
        f"Average metrics over {NUM_TRIALS} direct-client runs "
        f"for {NUM_QUERIES} queries per search type (fts, vector, hybrid)."
    )
    print(
        "| search | queries | runs | success_avg | elapsed_s_avg | qps_avg | "
        "p50_ms_avg | p95_ms_avg | p99_ms_avg | max_concurrency | seed | warmup_queries |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    # 三种检索类型共用同一份查询文本
    terms = get_query_terms("fts")
    rng = random.Random(args.seed)
    sampled_queries = rng.choices(terms, k=NUM_QUERIES)
    warmup_queries = [terms[j % len(terms)] for j in range(args.warmup_queries)]

    # 从 LanceDB 中随机抽取向量作为查询向量
    print(f"从 LanceDB 中随机抽取 {NUM_QUERIES} 条向量用于向量检索 ...")
    all_vectors = await fetch_random_vectors(table, NUM_QUERIES)
    while len(all_vectors) < NUM_QUERIES:
        all_vectors.extend(all_vectors[: NUM_QUERIES - len(all_vectors)])
    all_vectors = all_vectors[:NUM_QUERIES]
    query_vector_map = {q: all_vectors[i] for i, q in enumerate(sampled_queries)}
    for j, wq in enumerate(warmup_queries):
        if wq not in query_vector_map:
            query_vector_map[wq] = all_vectors[j % len(all_vectors)]
    print("向量准备完成。")

    # 关闭初始连接，后续每种检索类型使用独立连接
    db.close()
    del table, db
    gc.collect()

    for search_type in SEARCH_TYPES:
        # 每种检索类型创建独立的数据库连接，避免 Rust 端缓存累积导致性能退化
        db = await lancedb.connect_async(db_uri)
        table = await db.open_table("wikipedia_dedup")

        if search_type == "fts":
            query_runner = lambda query: run_lancedb_fts_query(table, query)
        elif search_type == "vector":
            query_runner = lambda query: run_lancedb_vector_query(
                table, query_vector_map[query]
            )
        else:
            query_runner = lambda query: run_lancedb_hybrid_query(
                table, query, query_vector_map[query]
            )

        concurrency = args.max_concurrency
        batch = None  # 不分批，一次性提交

        print(f"开始 {search_type} 压测（并发={concurrency}，batch={batch or 'all'}）...", flush=True)

        all_trial_metrics: list[dict[str, float]] = []
        for trial_idx in range(NUM_TRIALS):
            metrics = await run_single_trial(
                sampled_queries=sampled_queries,
                warmup_queries=warmup_queries,
                max_concurrency=concurrency,
                query_runner=query_runner,
                batch_size=batch,
            )
            all_trial_metrics.append(metrics)

        # 每种检索类型完成后关闭连接并清理内存，释放 Rust 端缓存
        db.close()
        del table, db
        gc.collect()

        avg = average_metrics(all_trial_metrics)
        print(
            f"| {search_type} | {NUM_QUERIES} | {NUM_TRIALS} | {avg['success']:.2f} | "
            f"{avg['elapsed_s']:.4f} | {avg['qps']:.2f} | {avg['p50_ms']:.2f} | "
            f"{avg['p95_ms']:.2f} | {avg['p99_ms']:.2f} | {concurrency} | "
            f"{args.seed} | {args.warmup_queries} |"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for deterministic query sampling",
    )
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument(
        "--warmup-queries",
        type=int,
        default=10,
        help="Number of warmup queries to run before each trial",
    )
    parsed_args = parser.parse_args()

    if parsed_args.max_concurrency <= 0:
        raise ValueError("--max-concurrency must be a positive integer")
    if parsed_args.warmup_queries < 0:
        raise ValueError("--warmup-queries must be >= 0")

    # 将控制台输出同时重定向到固定日志文件
    log_file = open(LOG_FILE, "a", encoding="utf-8")
    log_file.write(f"\n{'=' * 60}\n")
    log_file.write(f"Benchmark started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"{'=' * 60}\n")

    class TeeWriter:
        """同时写入控制台和日志文件的输出流。"""
        def __init__(self, console, file):
            self.console = console
            self.file = file

        def write(self, msg):
            self.console.write(msg)
            self.file.write(msg)

        def flush(self):
            self.console.flush()
            self.file.flush()

    original_stdout = sys.stdout
    sys.stdout = TeeWriter(original_stdout, log_file)
    try:
        asyncio.run(run_benchmark(parsed_args))
    finally:
        sys.stdout = original_stdout
        log_file.close()
        print(f"\n日志已保存到: {LOG_FILE}")
