"""Benchmark LanceDB using direct async client calls (no FastAPI)."""

import argparse
import asyncio
import os
import random
import sys
import warnings
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from time import perf_counter

# Suppress verbose Rust-side Lance warnings in benchmark output.
os.environ.setdefault("RUST_LOG", "error")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="lancedb")

import lancedb
from sentence_transformers import SentenceTransformer

try:
    from .config import Settings
except ImportError:
    from config import Settings

# 日志输出文件路径（固定位置）
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
EMBEDDING_DIM = 256
RESULT_COLUMNS = ["id", "title", "description", "country", "variety", "price", "points"]
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


async def run_lancedb_fts_query(table: lancedb.table.AsyncTable, query_text: str) -> bool:
    query = await table.search(
        query_text,
        query_type="fts",
        fts_columns=["description"],
    )
    result_table = await query.select(FTS_RESULT_COLUMNS).limit(10).to_arrow()
    return result_table.num_rows > 0


def precompute_vectors(
    model: SentenceTransformer,
    query_texts: list[str],
) -> dict[str, list[float]]:
    """预先批量编码所有查询词向量，排除 model.encode() 对压测计时的影响。"""
    vectors = {}
    for text in set(query_texts):
        vec = model.encode(
            f"search_query: {text.strip().lower()}",
            show_progress_bar=False,
            convert_to_numpy=True,
            truncate_dim=EMBEDDING_DIM,
        )
        if len(vec.shape) != 1 or vec.shape[0] != EMBEDDING_DIM:
            raise ValueError(
                f"Expected query embedding shape ({EMBEDDING_DIM},), got {vec.shape}"
            )
        vectors[text] = vec.astype("float32", copy=False).tolist()
    return vectors


async def run_lancedb_vector_query(
    table: lancedb.table.AsyncTable,
    query_vector: list[float],
) -> bool:
    """纯检索压测：只计算 LanceDB 检索耗时，不包含 model.encode() 时间。"""
    query = await table.search(
        query_vector,
        vector_column_name="vector",
        query_type="vector",
    )
    result_table = await (
        query.distance_type("cosine")
        .nprobes(10)
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
    """混合检索压测：同时使用 FTS 文本检索和向量检索，由 LanceDB 融合排序。
    直接使用底层 query() API 传入预计算向量，避免 search() 要求文本输入的限制。"""
    builder = (
        table.query()
        .nearest_to(query_vector)
        .column("vector")
        .distance_type("cosine")
        .nprobes(10)
        .nearest_to_text(query_text, columns=["description"])
        .select(RESULT_COLUMNS)
        .limit(10)
    )
    result_table = await builder.to_arrow()
    return result_table.num_rows > 0


async def run_single_trial(
    sampled_queries: list[str],
    warmup_queries: list[str],
    max_concurrency: int,
    query_runner,
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

    start_total = perf_counter()
    results = await asyncio.gather(*(timed_query(query) for query in sampled_queries))
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
    model = SentenceTransformer(settings.embedding_model_checkpoint)

    db_uri = Path(__file__).resolve().parent / settings.lancedb_dir
    db = await lancedb.connect_async(str(db_uri))
    table = await db.open_table("wines")

    print(
        f"Average metrics over {NUM_TRIALS} direct-client runs "
        f"for {NUM_QUERIES} queries per search type (fts, vector, hybrid)."
    )
    print(
        "| search | queries | runs | success_avg | elapsed_s_avg | qps_avg | "
        "p50_ms_avg | p95_ms_avg | p99_ms_avg | max_concurrency | seed | warmup_queries |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    # 三种检索类型共用同一份查询文本，保证压测对比的公平性
    terms = get_query_terms("fts")  # 所有类型都使用 keyword_terms.txt
    rng = random.Random(args.seed)
    sampled_queries = rng.choices(terms, k=NUM_QUERIES)
    warmup_queries = [terms[j % len(terms)] for j in range(args.warmup_queries)]

    # 预先批量编码所有查询词向量（向量检索和混合检索共用）
    all_texts = list(set(sampled_queries + warmup_queries))
    print(f"Pre-computing {len(all_texts)} unique query vectors (excluded from timing)...")
    vector_cache = precompute_vectors(model, all_texts)
    print("Vector pre-computation done.")

    for search_type in SEARCH_TYPES:
        if search_type == "fts":
            query_runner = lambda query: run_lancedb_fts_query(table, query)
        elif search_type == "vector":
            query_runner = lambda query: run_lancedb_vector_query(table, vector_cache[query])
        else:
            query_runner = lambda query: run_lancedb_hybrid_query(table, query, vector_cache[query])

        all_trial_metrics: list[dict[str, float]] = []
        for _ in range(NUM_TRIALS):
            all_trial_metrics.append(
                await run_single_trial(
                    sampled_queries=sampled_queries,
                    warmup_queries=warmup_queries,
                    max_concurrency=args.max_concurrency,
                    query_runner=query_runner,
                )
            )

        avg = average_metrics(all_trial_metrics)
        print(
            f"| {search_type} | {NUM_QUERIES} | {NUM_TRIALS} | {avg['success']:.2f} | "
            f"{avg['elapsed_s']:.4f} | {avg['qps']:.2f} | {avg['p50_ms']:.2f} | "
            f"{avg['p95_ms']:.2f} | {avg['p99_ms']:.2f} | {args.max_concurrency} | "
            f"{args.seed} | {args.warmup_queries} |"
        )

    db.close()


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
