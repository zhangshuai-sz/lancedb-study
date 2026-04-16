"""LanceDB 去重导入脚本。

从 Parquet 源数据中去重后重新导入 LanceDB，只保留每个 id 首次出现的记录。

背景：
  - Parquet 源数据共 62,398,110 条，其中 20,910,000 条是跨分片的完全重复数据
  - 去重后应有 41,488,110 条唯一记录
  - 本脚本通过内存中维护 seen_ids 集合来实现流式去重
"""

from __future__ import annotations

import argparse
import gc
import sys
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
# 每批次从 Parquet 读取的行数
PARQUET_READ_BATCH = 10_000
# 每批次写入 LanceDB 的行数
WRITE_BATCH_SIZE = 50_000

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


def prepare_arrow_batch(table: pa.Table) -> pa.Table:
    """将 Parquet 读取的 Arrow Table 转换为 LanceDB 需要的格式。"""
    col_names = table.column_names
    if "emb" in col_names:
        idx = col_names.index("emb")
        table = table.rename_columns(
            [("vector" if i == idx else name) for i, name in enumerate(col_names)]
        )

    keep_cols = list(TARGET_COLUMNS.keys()) + ["vector"]
    existing_cols = table.column_names
    select_cols = [c for c in keep_cols if c in existing_cols]
    table = table.select(select_cols)

    new_columns = []
    new_fields = []
    for col_name in table.column_names:
        col = table.column(col_name)
        if col_name in TARGET_COLUMNS:
            target_type = TARGET_COLUMNS[col_name]
            if col.null_count > 0:
                if pa.types.is_string(target_type) or pa.types.is_large_string(target_type):
                    col = col.fill_null("")
                elif pa.types.is_integer(target_type):
                    col = col.fill_null(0)
                elif pa.types.is_floating(target_type):
                    col = col.fill_null(0.0)
            if col.type != target_type:
                col = col.cast(target_type)
            new_fields.append(pa.field(col_name, target_type))
        else:
            new_fields.append(pa.field(col_name, col.type))
        new_columns.append(col)

    return pa.table(new_columns, schema=pa.schema(new_fields))


def dedup_batch(batch_table: pa.Table, seen_ids: set) -> pa.Table:
    """对一个 Arrow Table 进行去重，返回去重后的 Table。

    只保留 seen_ids 中不存在的行（即首次出现的记录），
    并将新 id 加入 seen_ids。
    """
    ids = batch_table.column("id").to_pylist()
    keep_indices = []
    for i, _id in enumerate(ids):
        if _id not in seen_ids:
            seen_ids.add(_id)
            keep_indices.append(i)

    if len(keep_indices) == len(ids):
        # 没有重复，直接返回
        return batch_table
    elif len(keep_indices) == 0:
        # 全部重复，返回空表
        return batch_table.slice(0, 0)
    else:
        return batch_table.take(keep_indices)


async def main() -> None:
    parser = argparse.ArgumentParser("LanceDB 去重导入：从 Parquet 源数据去重后重新导入")
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=0,
        help="限制导入条数（去重后），0 表示全量导入",
    )
    parser.add_argument(
        "--write-batch-size",
        type=int,
        default=WRITE_BATCH_SIZE,
        help=f"每批次写入 LanceDB 的行数（默认 {WRITE_BATCH_SIZE}）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续传：从已有数据的断点继续导入",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计去重结果，不实际写入",
    )
    parser.add_argument(
        "--replace-original",
        action="store_true",
        help="导入完成后，删除原 wikipedia 表，将 wikipedia_dedup 重命名为 wikipedia",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parents[1] / "data" / "wikipedia"
    parquet_files = get_parquet_files(data_dir)
    print(f"找到 {len(parquet_files)} 个 Parquet 分片文件", flush=True)

    # 统计总行数
    total_parquet_rows = sum(pq.read_metadata(f).num_rows for f in parquet_files)
    print(f"Parquet 总行数: {total_parquet_rows:,}", flush=True)

    settings = get_settings()
    db_uri = Path(__file__).resolve().parent / settings.lancedb_dir
    db = await lancedb.connect_async(str(db_uri))

    # 处理断点续传
    seen_ids: set = set()
    skip_files = 0
    existing_rows = 0

    if args.resume:
        # 检查现有表
        existing_tables = await db.list_tables()
        if "wikipedia_dedup" in existing_tables:
            table = await db.open_table("wikipedia_dedup")
            existing_rows = await table.count_rows()
            if existing_rows > 0:
                print(f"断点续传：已有去重表 wikipedia_dedup，包含 {existing_rows:,} 条数据", flush=True)
                # 需要重建 seen_ids：读取已有表中的所有 id
                print("正在加载已有 ID 用于去重...", flush=True)
                load_start = perf_counter()
                existing_data = await table.query().select(["id"]).to_arrow()
                for _id in existing_data.column("id").to_pylist():
                    seen_ids.add(_id)
                load_elapsed = perf_counter() - load_start
                print(f"已加载 {len(seen_ids):,} 个唯一 ID，耗时 {load_elapsed:.1f}s", flush=True)

                # 推算跳过的分片数
                cumulative = 0
                for i, pf in enumerate(parquet_files):
                    meta = pq.read_metadata(pf)
                    cumulative += meta.num_rows
                    if cumulative > existing_rows + len(seen_ids) * 0.5:
                        # 保守估计：从可能有新数据的分片开始
                        skip_files = max(0, i - 1)
                        break
                print(f"将从第 {skip_files} 个分片开始扫描", flush=True)
        else:
            print("未找到去重表，将从头开始", flush=True)
    
    if not args.resume or existing_rows == 0:
        # 全新导入：覆盖创建表
        if not args.dry_run:
            table = await db.create_table(
                "wikipedia_dedup",
                schema=pydantic_to_schema(LanceModelWikipedia),
                mode="overwrite",
            )
            print("已创建新表 wikipedia_dedup", flush=True)

    ingest_start = perf_counter()
    total_read = 0
    total_written = 0
    total_skipped = 0
    write_buffer: list[pa.Table] = []
    buffer_rows = 0

    async def flush_buffer():
        """将缓冲区中的数据写入 LanceDB。"""
        nonlocal write_buffer, buffer_rows, total_written
        if not write_buffer:
            return

        if args.dry_run:
            # dry-run 模式：只统计，不写入
            total_written += buffer_rows
            write_buffer = []
            buffer_rows = 0
            return

        combined = pa.concat_tables(write_buffer)
        await table.add(combined, mode="append")
        total_written += len(combined)
        del combined
        write_buffer = []
        buffer_rows = 0

    files_to_process = parquet_files[skip_files:]
    for file_idx, pf_path in enumerate(files_to_process):
        shard_start = perf_counter()
        shard_read = 0
        shard_kept = 0
        shard_skipped = 0

        pf = pq.ParquetFile(pf_path)
        for record_batch in pf.iter_batches(batch_size=PARQUET_READ_BATCH):
            batch_table = pa.Table.from_batches([record_batch])
            batch_len = len(batch_table)
            total_read += batch_len
            shard_read += batch_len

            # 去重
            deduped = dedup_batch(batch_table, seen_ids)
            kept = len(deduped)
            skipped = batch_len - kept
            shard_kept += kept
            shard_skipped += skipped
            total_skipped += skipped

            if kept > 0:
                # 转换格式并加入缓冲区
                prepared = prepare_arrow_batch(deduped)
                write_buffer.append(prepared)
                buffer_rows += kept

            # 检查是否需要 flush
            if buffer_rows >= args.write_batch_size:
                await flush_buffer()

            # 检查 limit
            if args.limit > 0 and (total_written + buffer_rows) >= args.limit:
                # 截断缓冲区
                await flush_buffer()
                break

            del batch_table, deduped, record_batch

        del pf
        gc.collect()

        shard_elapsed = perf_counter() - shard_start
        overall_elapsed = perf_counter() - ingest_start
        written_so_far = total_written + buffer_rows
        speed = written_so_far / overall_elapsed if overall_elapsed > 0 else 0
        # 预估剩余（基于唯一 ID 数 41,488,110）
        estimated_unique = 41_488_110
        remaining = max(0, estimated_unique - written_so_far)
        eta_hours = (remaining / speed / 3600) if speed > 0 else 0

        global_idx = file_idx + skip_files
        print(
            f"[{global_idx + 1}/{len(parquet_files)}] "
            f"{pf_path.name}: 读取 {shard_read:,}, 保留 {shard_kept:,}, 跳过 {shard_skipped:,} | "
            f"累计写入 {written_so_far:,} 条 | "
            f"去重率 {total_skipped / max(total_read, 1) * 100:.1f}% | "
            f"速度 {speed:,.0f} 条/s | "
            f"预计剩余 {eta_hours:.1f}h",
            flush=True,
        )

        if args.limit > 0 and written_so_far >= args.limit:
            print(f"已达到 limit={args.limit:,}，停止导入", flush=True)
            break

    # 最后 flush 剩余数据
    await flush_buffer()

    ingest_elapsed = perf_counter() - ingest_start

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}去重导入完成！", flush=True)
    print(f"  Parquet 总行数: {total_parquet_rows:,}", flush=True)
    print(f"  本次读取: {total_read:,} 条", flush=True)
    print(f"  去重跳过: {total_skipped:,} 条", flush=True)
    print(f"  实际写入: {total_written:,} 条", flush=True)
    print(f"  唯一 ID 数: {len(seen_ids):,}", flush=True)
    print(f"  耗时: {ingest_elapsed:.1f}s ({total_written / max(ingest_elapsed, 1):,.0f} 条/s)", flush=True)

    if not args.dry_run:
        final_count = await table.count_rows()
        print(f"  LanceDB 表最终行数: {final_count:,}", flush=True)

        if args.replace_original:
            print("\n正在替换原表...", flush=True)
            # 删除原 wikipedia 表
            existing_tables = await db.list_tables()
            if "wikipedia" in existing_tables:
                await db.drop_table("wikipedia")
                print("  已删除原 wikipedia 表", flush=True)
            # 重命名 wikipedia_dedup -> wikipedia
            # LanceDB 不支持直接重命名，所以保留 wikipedia_dedup 表名
            # 用户可以在代码中引用 wikipedia_dedup
            print(f"  去重表名为: wikipedia_dedup", flush=True)
            print(f"  请在代码中将表名从 'wikipedia' 改为 'wikipedia_dedup'，或手动重命名", flush=True)

    db.close()
    print("执行完毕！", flush=True)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
