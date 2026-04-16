"""
从 HuggingFace 下载 Cohere/wikipedia-2023-11-embed-multilingual-v3 数据集（英文子集），
并按分片保存为 Parquet 文件，供下游 ES / LanceDB 导入使用。

数据集字段：
  - _id: str          文档唯一标识
  - title: str         文章标题
  - text: str          段落文本
  - url: str           维基百科链接
  - wiki_id: int       维基百科文章 ID
  - views: float       页面浏览量
  - paragraph_id: int  段落序号
  - langs: int         多语言版本数
  - emb: list[float]   1024 维 Cohere embed-multilingual-v3.0 向量

使用方法：
  python convert.py                        # 下载全量英文数据（约 3500 万条）
  python convert.py --limit 1000000        # 只取前 100 万条
  python convert.py --lang en --shard-size 500000
  python convert.py --resume               # 断点续传，跳过已下载的分片
"""

import argparse
import os
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from rich import progress
from rich.console import Console


DATASET_NAME = "Cohere/wikipedia-2023-11-embed-multilingual-v3"
OUTPUT_DIR = Path(__file__).resolve().parent / "wikipedia"
# 单行迭代最大重试次数和初始退避秒数
MAX_ROW_RETRIES = 10
INITIAL_BACKOFF = 2

console = Console()


def _get_hf_token() -> str | None:
    """从环境变量获取 HuggingFace Token。"""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        console.print(f"[green]✓ 检测到 HF_TOKEN，将使用认证请求（速率限制更高）[/green]")
    else:
        console.print(
            "[yellow]⚠ 未设置 HF_TOKEN，使用匿名请求（容易被限速 503）。\n"
            "  建议: export HF_TOKEN=hf_xxxxx[/yellow]"
        )
    return token


def _resilient_iter(ds_iter, start_offset: int = 0):
    """
    包装数据集迭代器，为每一行添加重试 + 指数退避。
    当 HF Hub 返回 503 / timeout 时自动重试，而不是直接崩溃。
    """
    idx = 0
    iterator = iter(ds_iter)
    while True:
        retries = 0
        while True:
            try:
                row = next(iterator)
                break
            except StopIteration:
                return
            except Exception as e:
                retries += 1
                if retries > MAX_ROW_RETRIES:
                    console.print(f"[red]✗ 第 {idx} 行重试 {MAX_ROW_RETRIES} 次后仍失败: {e}[/red]")
                    raise
                backoff = min(INITIAL_BACKOFF * (2 ** (retries - 1)), 60)
                console.print(
                    f"[yellow]⚠ 第 {idx} 行迭代出错 (重试 {retries}/{MAX_ROW_RETRIES}，"
                    f"{backoff}s 后重试): {type(e).__name__}: {e}[/yellow]"
                )
                time.sleep(backoff)
                # 重新创建数据集迭代器并跳到当前位置
                try:
                    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                    ds_new = load_dataset(
                        DATASET_NAME, "en", split="train", streaming=True,
                        token=token,
                    )
                    iterator = iter(ds_new.skip(start_offset + idx))
                except Exception as reload_err:
                    console.print(f"[yellow]  重新加载数据集失败: {reload_err}，继续重试...[/yellow]")

        if idx == 0 and start_offset > 0:
            console.print(f"[green]✓ 从第 {start_offset} 条恢复迭代成功[/green]")
        idx += 1
        yield row


def _detect_resume_point(shard_size: int) -> tuple[int, int]:
    """
    检测已有的分片文件，返回 (起始分片索引, 已写入总行数)。
    只计算连续编号的分片（shard_00000, shard_00001, ...），遇到缺失则停止。
    """
    shard_idx = 0
    total_rows = 0
    while True:
        shard_path = OUTPUT_DIR / f"shard_{shard_idx:05d}.parquet"
        if not shard_path.exists():
            break
        # 读取实际行数
        meta = pq.read_metadata(shard_path)
        rows = meta.num_rows
        console.print(f"  [dim]已有分片 {shard_path.name}: {rows} 条[/dim]")
        total_rows += rows
        shard_idx += 1
    return shard_idx, total_rows


def download_and_convert(lang: str, limit: int, shard_size: int, resume: bool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    token = _get_hf_token()

    # 断点续传检测
    skip_rows = 0
    shard_idx = 0
    if resume:
        shard_idx, skip_rows = _detect_resume_point(shard_size)
        if skip_rows > 0:
            console.print(
                f"[green]✓ 断点续传: 已有 {shard_idx} 个分片 ({skip_rows} 条)，"
                f"从第 {skip_rows} 条继续下载[/green]"
            )
        else:
            console.print("[dim]未检测到已有分片，从头开始下载[/dim]")

    print(f"正在加载数据集 {DATASET_NAME} (lang={lang}) ...")
    ds = load_dataset(DATASET_NAME, lang, split="train", streaming=True, token=token)

    # 如果需要续传，跳过已下载的行
    if skip_rows > 0:
        console.print(f"跳过前 {skip_rows} 条已下载数据 ...")
        ds = ds.skip(skip_rows)

    buffer: list[dict] = []
    total_written = skip_rows

    remaining = (limit - skip_rows) if limit > 0 else None

    with progress.Progress(
        "[progress.description]{task.description}",
        progress.BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        progress.TimeRemainingColumn(),
        progress.TimeElapsedColumn(),
        console=console,
    ) as prog:
        desc = f"下载中 (limit={limit if limit > 0 else '全量'}, 已有={skip_rows})"
        task = prog.add_task(desc, total=remaining)

        try:
            for i, row in enumerate(_resilient_iter(ds, start_offset=skip_rows)):
                if remaining is not None and i >= remaining:
                    break

                buffer.append({
                    "id": row["_id"],
                    "title": row["title"],
                    "text": row["text"],
                    "url": row.get("url", ""),
                    "wiki_id": row.get("wiki_id", 0),
                    "views": row.get("views", 0.0),
                    "paragraph_id": row.get("paragraph_id", 0),
                    "langs": row.get("langs", 0),
                    "emb": row["emb"],
                })

                if len(buffer) >= shard_size:
                    _flush_shard(buffer, shard_idx)
                    total_written += len(buffer)
                    shard_idx += 1
                    buffer.clear()

                prog.update(task, advance=1)

        except KeyboardInterrupt:
            console.print("\n[yellow]⚠ 用户中断，正在保存已下载的数据...[/yellow]")

    # 写入剩余数据（包括中断时的部分数据）
    if buffer:
        _flush_shard(buffer, shard_idx)
        total_written += len(buffer)
        shard_idx += 1

    print(f"完成！共写入 {total_written} 条数据，{shard_idx} 个分片，保存在 {OUTPUT_DIR}")


def _flush_shard(buffer: list[dict], shard_idx: int) -> None:
    """将缓冲区数据写入一个 Parquet 分片文件。"""
    table = pa.Table.from_pylist(buffer)
    out_path = OUTPUT_DIR / f"shard_{shard_idx:05d}.parquet"
    pq.write_table(table, out_path, compression="zstd")
    print(f"  写入分片 {out_path.name} ({len(buffer)} 条)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("下载 Cohere Wikipedia 数据集并转为 Parquet 分片")
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="语言代码，默认 en（英文）",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=0,
        help="限制下载条数，0 表示全量下载",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=500_000,
        help="每个 Parquet 分片的行数",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="断点续传：检测已有分片并从断点继续下载",
    )
    args = parser.parse_args()
    download_and_convert(args.lang, args.limit, args.shard_size, args.resume)