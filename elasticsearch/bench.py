"""Benchmark Elasticsearch using direct async client calls (no FastAPI)."""

import argparse
import asyncio
import random
from functools import lru_cache
from pathlib import Path
from time import perf_counter

from elasticsearch import AsyncElasticsearch

try:
    from .config import Settings
except ImportError:
    from config import Settings

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
    client: AsyncElasticsearch,
    index_alias: str,
    n: int,
    seed: int,
) -> list[list[float]]:
    """从 ES 索引中随机抽取 n 条文档的向量，用作向量检索的查询向量。"""
    response = await client.search(
        index=index_alias,
        size=n,
        query={
            "function_score": {
                "query": {"match_all": {}},
                "random_score": {"seed": seed, "field": "_seq_no"},
            }
        },
        _source=["vector"],
    )
    vectors = []
    for hit in response["hits"]["hits"]:
        vec = hit["_source"].get("vector")
        if vec:
            vectors.append(vec)
    if not vectors:
        raise RuntimeError("无法从 ES 中获取向量数据，请确认数据已导入")
    return vectors


async def run_es_fts_query(client: AsyncElasticsearch, index_alias: str, query_text: str) -> bool:
    response = await client.search(
        index=index_alias,
        size=10,
        query={
            "match": {
                "text": {
                    "query": query_text,
                }
            }
        },
        _source=RESULT_COLUMNS,
    )
    return bool(response["hits"].get("hits"))


async def run_es_vector_query(
    client: AsyncElasticsearch,
    index_alias: str,
    query_vector: list[float],
) -> bool:
    response = await client.search(
        index=index_alias,
        knn={
            "field": "vector",
            "query_vector": query_vector,
            "k": 10,
            "num_candidates": 100,
        },
        _source=RESULT_COLUMNS,
    )
    return bool(response["hits"].get("hits"))


async def run_es_hybrid_query(
    client: AsyncElasticsearch,
    index_alias: str,
    query_text: str,
    query_vector: list[float],
) -> bool:
    response = await client.search(
        index=index_alias,
        knn={
            "field": "vector",
            "query_vector": query_vector,
            "k": 10,
            "num_candidates": 100,
        },
        query={
            "match": {
                "text": {
                    "query": query_text,
                }
            }
        },
        size=10,
        _source=RESULT_COLUMNS,
    )
    return bool(response["hits"].get("hits"))


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
    index_alias = settings.elastic_index_alias

    client = AsyncElasticsearch(
        f"http://{settings.elastic_url}:{settings.elastic_port}",
        basic_auth=(settings.elastic_user, settings.elastic_password),
        request_timeout=60,
        max_retries=3,
        retry_on_timeout=True,
        verify_certs=False,
    )

    print(
        f"Averaged metrics over {NUM_TRIALS} direct-client runs "
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

    # 从 ES 中随机抽取向量作为查询向量（向量检索和混合检索共用）
    print(f"从 ES 中随机抽取 {NUM_QUERIES} 条向量用于向量检索 ...")
    all_vectors = await fetch_random_vectors(client, index_alias, NUM_QUERIES, args.seed)
    # 如果抽取的向量不够，循环复用
    while len(all_vectors) < NUM_QUERIES:
        all_vectors.extend(all_vectors[: NUM_QUERIES - len(all_vectors)])
    all_vectors = all_vectors[:NUM_QUERIES]
    # 为每个查询文本分配一个向量
    query_vector_map = {q: all_vectors[i] for i, q in enumerate(sampled_queries)}
    # warmup 也需要向量
    for j, wq in enumerate(warmup_queries):
        if wq not in query_vector_map:
            query_vector_map[wq] = all_vectors[j % len(all_vectors)]
    print("向量准备完成。")

    for search_type in SEARCH_TYPES:
        if search_type == "fts":
            query_runner = lambda query: run_es_fts_query(client, index_alias, query)
        elif search_type == "vector":
            query_runner = lambda query: run_es_vector_query(
                client, index_alias, query_vector_map[query]
            )
        else:
            query_runner = lambda query: run_es_hybrid_query(
                client, index_alias, query, query_vector_map[query]
            )

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

    await client.close()


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

    asyncio.run(run_benchmark(parsed_args))
