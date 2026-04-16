"""创建 LanceDB FTS 和 IVF_PQ 索引（导入数据后执行）。"""

import argparse
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import lancedb
from config import Settings
from lancedb.index import FTS, IvfPq


@lru_cache()
def get_settings() -> Settings:
    return Settings()


async def main() -> None:
    parser = argparse.ArgumentParser("创建 LanceDB 索引")
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=16384,
        help="IVF 分区数（数据量大时建议增大）",
    )
    parser.add_argument(
        "--num-sub-vectors",
        type=int,
        default=64,
        help="PQ 子向量数",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="替换已有索引",
    )
    args = parser.parse_args()

    settings = get_settings()
    db_uri = Path(__file__).resolve().parent / settings.lancedb_dir
    db = await lancedb.connect_async(str(db_uri))
    table = await db.open_table("wikipedia_dedup")

    fts_start = perf_counter()
    await table.create_index(
        "text",
        config=FTS(),
        replace=True,
    )
    fts_elapsed = perf_counter() - fts_start
    print(f"FTS 索引创建完成，耗时 {fts_elapsed:.4f}s")

    ivfpq_start = perf_counter()
    await table.create_index(
        "vector",
        config=IvfPq(
            distance_type="cosine",
            num_partitions=args.num_partitions,
            num_sub_vectors=args.num_sub_vectors,
        ),
        replace=True,
    )
    ivfpq_elapsed = perf_counter() - ivfpq_start
    print(
        f"IVF_PQ 索引创建完成，耗时 {ivfpq_elapsed:.4f}s "
        f"(num_partitions={args.num_partitions}, num_sub_vectors={args.num_sub_vectors})"
    )

    db.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
