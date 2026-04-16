"""FastAPI app to serve LanceDB Wikipedia search endpoints."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import lru_cache
import os
from pathlib import Path
import warnings

# 抑制 Rust 端的 Lance 警告
os.environ.setdefault("RUST_LOG", "error")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="lancedb")

import lancedb
from fastapi import FastAPI, HTTPException, Query, Request

try:
    from .config import Settings
    from .schemas.wikipedia import SearchResult
except ImportError:
    from config import Settings
    from schemas.wikipedia import SearchResult

RESULT_COLUMNS = ["id", "title", "text", "url", "wiki_id", "views", "paragraph_id", "langs"]
FTS_RESULT_COLUMNS = [*RESULT_COLUMNS, "_score"]
VECTOR_RESULT_COLUMNS = [*RESULT_COLUMNS, "_distance"]
EMBEDDING_DIM = 1024


@lru_cache()
def get_settings() -> Settings:
    return Settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Async context manager for LanceDB connection."""
    settings = get_settings()
    db_uri = Path(__file__).resolve().parent / settings.lancedb_dir
    app.db = await lancedb.connect_async(str(db_uri))
    app.table = await app.db.open_table("wikipedia_dedup")
    print("成功连接到 LanceDB")
    yield
    app.db.close()
    print("已关闭 LanceDB 连接")


app = FastAPI(
    title="REST API for Wikipedia search on LanceDB",
    description="基于 Cohere Wikipedia 数据集的 LanceDB 检索 API",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {
        "message": "REST API for querying LanceDB database of Wikipedia articles"
    }


async def _fts_search(request: Request, terms: str) -> list[dict[str, object]] | None:
    query = await request.app.table.search(
        terms,
        query_type="fts",
        fts_columns=["text"],
    )
    result_table = await query.select(FTS_RESULT_COLUMNS).limit(10).to_arrow()
    if result_table.num_rows == 0:
        return None
    return result_table.to_pylist()


async def _vector_search(request: Request, query_vector: list[float]) -> list[dict[str, object]] | None:
    query = await request.app.table.search(
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
    if result_table.num_rows == 0:
        return None
    return result_table.to_pylist()


# --- Endpoints ---


@app.get(
    "/fts_search",
    response_model=list[SearchResult],
    response_description="通过全文关键词搜索 Wikipedia 文章",
)
async def fts_search(
    request: Request,
    query: str = Query(description="搜索关键词"),
) -> list[SearchResult]:
    result = await _fts_search(request, query)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"未找到与 '{query}' 相关的文章",
        )
    return result


@app.get(
    "/vector_search",
    response_model=list[SearchResult],
    response_description="通过语义向量搜索 Wikipedia 文章",
)
async def vector_search(
    request: Request,
    query: str = Query(description="搜索文本（当前版本需要外部提供向量）"),
) -> list[SearchResult]:
    # 注意：由于数据集使用 Cohere 模型生成的向量，
    # 实际生产中应使用 Cohere API 对查询文本做 embedding。
    raise HTTPException(
        status_code=501,
        detail="向量搜索需要 Cohere embedding API 对查询文本编码，请使用 bench.py 中的预计算向量进行测试",
    )