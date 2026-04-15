"""
FastAPI app to serve Wikipedia search endpoints on Elasticsearch
"""
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query, Request

from elasticsearch import AsyncElasticsearch

try:
    from .config import Settings
    from .schemas.wikipedia import SearchResult
except ImportError:
    from config import Settings
    from schemas.wikipedia import SearchResult

EMBEDDING_DIM = 1024


@lru_cache()
def get_settings():
    return Settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Async context manager for Elasticsearch connection."""
    settings = get_settings()

    username = settings.elastic_user
    password = settings.elastic_password
    port = settings.elastic_port
    service = settings.elastic_url
    elastic_client = AsyncElasticsearch(
        f"http://{service}:{port}",
        basic_auth=(username, password),
        request_timeout=60,
        max_retries=3,
        retry_on_timeout=True,
        verify_certs=False,
    )
    app.client = elastic_client
    app.index_alias = settings.elastic_index_alias
    print("成功连接到 Elasticsearch")
    yield
    await elastic_client.close()
    print("已关闭 Elasticsearch 连接")


app = FastAPI(
    title="REST API for Wikipedia search on Elasticsearch",
    description="基于 Cohere Wikipedia 数据集的 Elasticsearch 检索 API",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "REST API for querying Elasticsearch index of Wikipedia articles"
    }


# --- Search functions ---


async def _fts_search(request: Request, query: str) -> list[SearchResult] | None:
    response = await request.app.client.search(
        index=request.app.index_alias,
        size=10,
        query={
            "match": {
                "text": {
                    "query": query,
                }
            }
        },
        _source=["id", "title", "text", "url", "wiki_id", "views", "paragraph_id", "langs"],
    )
    result = response["hits"].get("hits")
    if result:
        return [item["_source"] for item in result]
    else:
        return None


async def _vector_search(request: Request, query: str, query_vector: list[float]) -> list[SearchResult] | None:
    response = await request.app.client.search(
        index=request.app.index_alias,
        knn={
            "field": "vector",
            "query_vector": query_vector,
            "k": 10,
            "num_candidates": 100,
        },
        _source=["id", "title", "text", "url", "wiki_id", "views", "paragraph_id", "langs"],
    )
    result = response["hits"].get("hits")
    if result:
        return [item["_source"] for item in result]
    else:
        return None


# --- Endpoints ---


@app.get(
    "/fts_search",
    response_model=list[SearchResult],
    response_description="通过全文关键词搜索 Wikipedia 文章",
)
async def fts_search(
    request: Request,
    query: str = Query(description="搜索关键词"),
) -> list[SearchResult] | None:
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
    response_description="通过语义向量搜索 Wikipedia 文章（需要传入预计算的查询向量）",
)
async def vector_search(
    request: Request,
    query: str = Query(description="搜索文本（当前版本需要外部提供向量，此参数仅用于展示）"),
) -> list[SearchResult] | None:
    # 注意：由于数据集使用 Cohere 模型生成的向量，
    # 实际生产中应使用 Cohere API 对查询文本做 embedding。
    # 这里暂时使用 FTS 作为 fallback。
    raise HTTPException(
        status_code=501,
        detail="向量搜索需要 Cohere embedding API 对查询文本编码，请使用 bench.py 中的预计算向量进行测试",
    )