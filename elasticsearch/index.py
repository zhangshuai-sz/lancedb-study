"""将 Wikipedia Parquet 分片数据导入 Elasticsearch（使用数据集自带的 1024 维向量）。

优化要点：
  - 使用 ParquetFile.iter_batches() 流式读取，避免一次性加载整个分片到内存
  - 去掉逐行 Pydantic 验证，直接从 Arrow RecordBatch 提取字段（6000 万行验证开销巨大）
  - 使用生成器模式，避免中间列表的内存开销
  - 支持 --resume 断点续传，跳过已导入的分片
"""

import argparse
import gc
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import pyarrow.parquet as pq
from config import Settings
from elasticsearch.helpers import BulkIndexError
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from elasticsearch import Elasticsearch, helpers

JsonBlob = dict[str, Any]
EMBEDDING_DIM = 1024
# iter_batches 每批读取的行数（控制内存峰值，10000 行 × 1024 维 ≈ 40MB Arrow 内存）
PARQUET_READ_BATCH = 10_000

console = Console()


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def get_parquet_files(data_dir: Path) -> list[Path]:
    """获取所有 Parquet 分片文件，按文件名排序。"""
    files = sorted(data_dir.glob("shard_*.parquet"))
    if not files:
        raise FileNotFoundError(f"在 {data_dir} 下未找到 shard_*.parquet 文件")
    return files


def iter_actions_from_parquet(
    parquet_files: list[Path],
    index_alias: str,
    limit: int = 0,
    skip_files: int = 0,
) -> Iterator[JsonBlob]:
    """
    流式生成 ES bulk action，逐分片、逐批次读取 Parquet 文件。

    使用 ParquetFile.iter_batches() 避免一次性加载整个分片到内存。
    每次只在内存中保留一个小批次（PARQUET_READ_BATCH 行）。
    """
    total_yielded = 0
    files_to_process = parquet_files[skip_files:]

    for file_idx, pf in enumerate(files_to_process):
        if 0 < limit <= total_yielded:
            break

        parquet_file = pq.ParquetFile(pf)

        for batch in parquet_file.iter_batches(batch_size=PARQUET_READ_BATCH):
            if 0 < limit <= total_yielded:
                break

            # 将 Arrow RecordBatch 转为列式字典，比 to_pylist() 更高效
            ids = batch.column("id").to_pylist()
            titles = batch.column("title").to_pylist()
            texts = batch.column("text").to_pylist()
            urls = batch.column("url").to_pylist()
            wiki_ids = batch.column("wiki_id").to_pylist()
            views_col = batch.column("views").to_pylist()
            paragraph_ids = batch.column("paragraph_id").to_pylist()
            langs_col = batch.column("langs").to_pylist()
            embs = batch.column("emb").to_pylist()

            for i in range(len(ids)):
                if 0 < limit <= total_yielded:
                    break
                yield {
                    "_index": index_alias,
                    "_id": ids[i],
                    "id": ids[i],
                    "title": titles[i],
                    "text": texts[i],
                    "url": urls[i] or "",
                    "wiki_id": wiki_ids[i] or 0,
                    "views": views_col[i] or 0.0,
                    "paragraph_id": paragraph_ids[i] or 0,
                    "langs": langs_col[i] or 0,
                    "vector": embs[i],
                }
                total_yielded += 1

        # 每个分片处理完后主动回收内存
        gc.collect()


def get_elastic_client(settings: Settings) -> Elasticsearch:
    return Elasticsearch(
        f"http://{settings.elastic_url}:{settings.elastic_port}",
        basic_auth=(settings.elastic_user, settings.elastic_password),
        request_timeout=300,
        max_retries=3,
        retry_on_timeout=True,
        verify_certs=False,
    )


def create_index_if_needed(
    client: Elasticsearch, index_alias: str, mappings_path: Path
) -> None:
    import srsly

    elastic_config = dict(srsly.read_json(mappings_path))
    mappings = elastic_config.get("mappings")
    settings = elastic_config.get("settings")

    if client.indices.exists_alias(name=index_alias):
        console.print(f"[green]发现索引别名 `{index_alias}`，跳过创建[/green]")
        return

    index_name = f"{index_alias}-1"
    console.print(f"未找到别名 `{index_alias}`，正在创建索引 `{index_name}`")
    client.indices.create(index=index_name, mappings=mappings, settings=settings)
    client.indices.put_alias(index=index_name, name=index_alias)


def drop_alias_indices(client: Elasticsearch, index_alias: str) -> None:
    if not client.indices.exists_alias(name=index_alias):
        return
    indices = list(client.indices.get_alias(name=index_alias).keys())
    for index_name in indices:
        console.print(f"正在删除索引 `{index_name}` (别名 `{index_alias}`)")
        client.indices.delete(index=index_name)


def count_total_rows(parquet_files: list[Path], limit: int = 0) -> int:
    """快速统计总行数（只读元数据，不加载数据）。"""
    total = 0
    for pf in parquet_files:
        meta = pq.read_metadata(pf)
        total += meta.num_rows
    if 0 < limit < total:
        return limit
    return total


def count_rows_in_files(parquet_files: list[Path]) -> list[int]:
    """返回每个分片文件的行数列表。"""
    return [pq.read_metadata(pf).num_rows for pf in parquet_files]


def detect_resume_point(client: Elasticsearch, index_alias: str) -> int:
    """查询 ES 中已有的文档数，用于断点续传。"""
    try:
        resp = client.count(index=index_alias)
        return resp["count"]
    except Exception:
        return 0


def main(args: argparse.Namespace) -> None:
    settings = get_settings()
    index_alias = settings.elastic_index_alias

    data_dir = Path(__file__).resolve().parents[1] / "data" / "wikipedia"
    parquet_files = get_parquet_files(data_dir)
    console.print(f"找到 [bold]{len(parquet_files)}[/bold] 个 Parquet 分片文件")

    # 统计每个分片的行数
    rows_per_file = count_rows_in_files(parquet_files)
    total_rows_all = sum(rows_per_file)
    console.print(f"数据总量: [bold]{total_rows_all:,}[/bold] 条")

    client = get_elastic_client(settings)
    if not client.ping():
        raise RuntimeError("无法连接到 Elasticsearch")

    mappings_path = Path(__file__).resolve().parent / "mapping" / "mapping.json"
    if args.recreate_index:
        drop_alias_indices(client, index_alias)
    create_index_if_needed(client, index_alias, mappings_path)

    # 断点续传：根据 ES 已有文档数，计算应跳过的分片数
    skip_files = 0
    skip_rows = 0
    if args.resume:
        existing_count = detect_resume_point(client, index_alias)
        if existing_count > 0:
            # 根据已有文档数推算已完成的分片
            cumulative = 0
            for i, row_count in enumerate(rows_per_file):
                cumulative += row_count
                if cumulative >= existing_count:
                    # 如果刚好等于累计值，说明前 i+1 个分片已完成
                    if cumulative == existing_count:
                        skip_files = i + 1
                        skip_rows = existing_count
                    else:
                        # 不完整的分片，从该分片重新开始（ES 的 _id 会自动去重）
                        skip_files = i
                        skip_rows = cumulative - row_count
                    break
            console.print(
                f"[green]断点续传: ES 已有 {existing_count:,} 条，"
                f"跳过前 {skip_files} 个分片 ({skip_rows:,} 条)，"
                f"从第 {skip_files} 个分片继续[/green]"
            )

    # 计算实际需要导入的行数
    remaining_rows = total_rows_all - skip_rows
    if args.limit > 0:
        remaining_rows = min(remaining_rows, args.limit)
    console.print(f"本次预计导入: [bold]{remaining_rows:,}[/bold] 条")

    # 导入前确保 refresh_interval 为 -1（提升写入性能）
    actual_indices = list(client.indices.get_alias(name=index_alias).keys())
    for idx_name in actual_indices:
        client.indices.put_settings(
            index=idx_name, settings={"refresh_interval": "-1"}
        )
    console.print("[dim]已设置 refresh_interval=-1 以提升写入性能[/dim]")

    # 流式生成 bulk actions
    actions = iter_actions_from_parquet(
        parquet_files, index_alias, limit=args.limit, skip_files=skip_files
    )

    ingest_start = perf_counter()
    indexed_count = 0
    error_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        redirect_stdout=True,
        redirect_stderr=True,
    ) as prog:
        task = prog.add_task("导入 Elasticsearch", total=remaining_rows)

        try:
            for ok, info in helpers.streaming_bulk(
                client,
                actions,
                chunk_size=args.chunksize,
                raise_on_error=False,
                raise_on_exception=False,
            ):
                if ok:
                    indexed_count += 1
                else:
                    error_count += 1
                    if error_count <= 5:
                        console.print(f"[red]导入错误: {info}[/red]")

                if indexed_count % args.chunksize == 0:
                    prog.update(task, completed=indexed_count)

        except KeyboardInterrupt:
            console.print("\n[yellow]⚠ 用户中断，已导入的数据不会丢失[/yellow]")
        except Exception as e:
            console.print(f"\n[red]导入异常: {e}[/red]")

    # 最终更新进度条
    prog_elapsed = perf_counter() - ingest_start

    console.print(
        f"\n[bold green]导入完成！[/bold green]\n"
        f"  成功: {indexed_count:,} 条\n"
        f"  失败: {error_count:,} 条\n"
        f"  耗时: {prog_elapsed:.1f}s "
        f"({indexed_count / max(prog_elapsed, 1):.0f} docs/s)"
    )

    # 导入完成后恢复刷新间隔
    console.print("正在恢复 refresh_interval 为 1s ...")
    for idx_name in actual_indices:
        client.indices.put_settings(
            index=idx_name, settings={"refresh_interval": "1s"}
        )
    # 强制刷新使数据可搜索
    for idx_name in actual_indices:
        client.indices.refresh(index=idx_name)
    console.print("[green]刷新间隔已恢复，索引已刷新[/green]")

    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("将 Wikipedia Parquet 数据导入 Elasticsearch")
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
        default=2000,
        help="每批次发送给 ES 的文档数",
    )
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="删除已有索引后重新创建",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续传：根据 ES 已有文档数跳过已导入的分片",
    )
    parsed_args = parser.parse_args()

    main(parsed_args)