"""将 Wikipedia Parquet 分片数据导入 LanceDB（使用数据集自带的 1024 维向量）。

优化版本：使用 PyArrow 原生操作，避免 to_pylist() 的巨大内存开销。
"""

from __future__ import annotations

import argparse
import gc
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Optional

import lancedb
import pyarrow as pa
import pyarrow.parquet as pq
from config import Settings
from lancedb.pydantic import pydantic_to_schema
from schemas.wikipedia import LanceModelWikipedia

EMBEDDING_DIM = 1024

# LanceDB 表需要的列及其目标类型
TARGET_COLUMNS = {
    "id": pa.utf8(),
    "title": pa.utf8(),
    "text": pa.utf8(),
    "url": pa.utf8(),
    "wiki_id": pa.int64(),
    "views": pa.float64(),
    "paragraph_id": pa.int64(),
    "langs": pa.int64(),
}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def get_parquet_files(data_dir: Path) -> list[Path]:
    """获取所有 Parquet 分片文件，按文件名排序。"""
    files = sorted(data_dir.glob("shard_*.parquet"))
    if not files:
        raise FileNotFoundError(f"在 {data_dir} 下未找到 shard_*.parquet 文件")
    return files


def count_total_rows(parquet_files: list[Path], limit: int = 0) -> int:
    """快速统计总行数（用于进度条）。"""
    total = 0
    for pf in parquet_files:
        meta = pq.read_metadata(pf)
        total += meta.num_rows
    if 0 < limit < total:
        return limit
    return total


def prepare_arrow_batch(table: pa.Table) -> pa.Table:
    """将 Parquet 读取的 Arrow Table 转换为 LanceDB 需要的格式。

    主要操作：
    1. 将 'emb' 列重命名为 'vector'
    2. 只保留目标列
    3. 处理空值填充
    """
    # 重命名 emb -> vector
    col_names = table.column_names
    if "emb" in col_names:
        idx = col_names.index("emb")
        table = table.rename_columns(
            [("vector" if i == idx else name) for i, name in enumerate(col_names)]
        )

    # 只保留需要的列
    keep_cols = list(TARGET_COLUMNS.keys()) + ["vector"]
    existing_cols = table.column_names
    select_cols = [c for c in keep_cols if c in existing_cols]
    table = table.select(select_cols)

    # 填充空值并转换类型
    new_columns = []
    new_fields = []
    for col_name in table.column_names:
        col = table.column(col_name)
        if col_name in TARGET_COLUMNS:
            target_type = TARGET_COLUMNS[col_name]
            # 填充空值
            if col.null_count > 0:
                if pa.types.is_string(target_type) or pa.types.is_large_string(target_type):
                    col = col.fill_null("")
                elif pa.types.is_integer(target_type):
                    col = col.fill_null(0)
                elif pa.types.is_floating(target_type):
                    col = col.fill_null(0.0)
            # 类型转换
            if col.type != target_type:
                col = col.cast(target_type)
            new_fields.append(pa.field(col_name, target_type))
        else:
            # vector 列保持原样
            new_fields.append(pa.field(col_name, col.type))
        new_columns.append(col)

    return pa.table(new_columns, schema=pa.schema(new_fields))


async def get_or_create_table(
    db: lancedb.db.AsyncConnection,
    table_name: str,
    overwrite: bool,
) -> lancedb.table.AsyncTable:
    if overwrite:
        return await db.create_table(
            table_name,
            schema=pydantic_to_schema(LanceModelWikipedia),
            mode="overwrite",
        )

    existing_tables = await db.table_names()
    if table_name in existing_tables:
        table = await db.open_table(table_name)
        schema = await table.schema()
        vector_type = schema.field("vector").type if "vector" in schema.names else None
        existing_dim = getattr(vector_type, "list_size", None)
        if existing_dim is not None and existing_dim != EMBEDDING_DIM:
            raise ValueError(
                f"已有表 '{table_name}' 的向量维度为 {existing_dim}，"
                f"但当前导入需要 {EMBEDDING_DIM} 维。"
                "请使用 --overwrite 重建表。"
            )
        return table

    return await db.create_table(
        table_name,
        schema=pydantic_to_schema(LanceModelWikipedia),
        mode="create",
    )


async def ingest_shard(
    table: lancedb.table.AsyncTable,
    parquet_path: Path,
    batch_size: int,
    limit_remaining: int = 0,
) -> int:
    """导入单个 Parquet 分片文件，使用 iter_batches 控制内存。

    返回实际导入的行数。
    """
    pf = pq.ParquetFile(parquet_path)
    ingested = 0

    for record_batch in pf.iter_batches(batch_size=batch_size):
        if 0 < limit_remaining <= ingested:
            break

        # 转为 Table 以便操作
        batch_table = pa.Table.from_batches([record_batch])

        # 如果有 limit，截断
        if limit_remaining > 0:
            remaining = limit_remaining - ingested
            if len(batch_table) > remaining:
                batch_table = batch_table.slice(0, remaining)

        # 转换为 LanceDB 格式
        prepared = prepare_arrow_batch(batch_table)

        # 写入 LanceDB
        await table.add(prepared, mode="append")
        ingested += len(prepared)

        # 释放内存
        del batch_table, prepared, record_batch

    del pf
    gc.collect()
    return ingested


async def main() -> None:
    parser = argparse.ArgumentParser("将 Wikipedia Parquet 数据导入 LanceDB")
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=0,
        help="限制导入条数，0 表示全量导入",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=10000,
        help="每批次导入的文档数（Arrow batch size）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有的表",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续传：跳过已导入的分片文件，从断点继续导入",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parents[1] / "data" / "wikipedia"
    parquet_files = get_parquet_files(data_dir)
    print(f"找到 {len(parquet_files)} 个 Parquet 分片文件", flush=True)

    settings = get_settings()
    db_uri = Path(__file__).resolve().parent / settings.lancedb_dir
    db = await lancedb.connect_async(str(db_uri))
    table = await get_or_create_table(db, "wikipedia", overwrite=args.overwrite)

    # 断点续传：根据已有行数跳过已导入的分片
    skip_count = 0
    if args.resume and not args.overwrite:
        existing_rows = await table.count_rows()
        if existing_rows > 0:
            cumulative = 0
            for pf in parquet_files:
                meta = pq.read_metadata(pf)
                cumulative += meta.num_rows
                if cumulative <= existing_rows:
                    skip_count += 1
                else:
                    break
            parquet_files = parquet_files[skip_count:]
            print(
                f"断点续传：已有 {existing_rows:,} 条数据，跳过前 {skip_count} 个分片，"
                f"从 {parquet_files[0].name if parquet_files else '无'} 开始",
                flush=True,
            )
        else:
            print("表为空，从头开始导入", flush=True)

    total_rows = count_total_rows(parquet_files, args.limit)
    print(f"本次预计导入: {total_rows:,} 条数据", flush=True)

    if not parquet_files:
        print("没有需要导入的分片文件，退出。", flush=True)
        db.close()
        return

    ingest_start = perf_counter()
    total_ingested = 0
    limit_remaining = args.limit if args.limit > 0 else 0

    for i, pf_path in enumerate(parquet_files):
        shard_start = perf_counter()
        shard_limit = (limit_remaining - total_ingested) if limit_remaining > 0 else 0

        if limit_remaining > 0 and total_ingested >= limit_remaining:
            break

        ingested = await ingest_shard(table, pf_path, args.chunksize, shard_limit)
        total_ingested += ingested
        shard_elapsed = perf_counter() - shard_start
        overall_elapsed = perf_counter() - ingest_start

        # 计算速度和预估剩余时间
        speed = total_ingested / overall_elapsed if overall_elapsed > 0 else 0
        remaining_rows = total_rows - total_ingested
        eta_seconds = remaining_rows / speed if speed > 0 else 0
        eta_hours = eta_seconds / 3600

        print(
            f"[{i + 1 + skip_count}/{len(parquet_files) + skip_count}] "
            f"{pf_path.name}: +{ingested:,} 条, "
            f"耗时 {shard_elapsed:.1f}s | "
            f"累计 {total_ingested:,}/{total_rows:,} 条 "
            f"({total_ingested / total_rows * 100:.1f}%) | "
            f"速度 {speed:,.0f} 条/s | "
            f"预计剩余 {eta_hours:.1f}h",
            flush=True,
        )

    ingest_elapsed = perf_counter() - ingest_start
    final_count = await table.count_rows()
    print(
        f"\n导入完成！本次导入 {total_ingested:,} 条，耗时 {ingest_elapsed:.2f}s "
        f"(平均 {total_ingested / ingest_elapsed:,.0f} 条/s)",
        flush=True,
    )
    print(f"LanceDB 表总行数: {final_count:,}", flush=True)

    db.close()
    print("执行完毕！", flush=True)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())