"""LanceDB 端的 Wikipedia 文档数据模型。"""

from typing import Optional

from pydantic import BaseModel, ConfigDict

from lancedb.pydantic import LanceModel, Vector


class Wikipedia(BaseModel):
    """Cohere/wikipedia-2023-11-embed-multilingual-v3 数据集的文档模型（验证用）。"""

    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
        extra="allow",
        str_strip_whitespace=True,
    )

    id: str
    title: str
    text: str
    url: Optional[str] = ""
    wiki_id: Optional[int] = 0
    views: Optional[float] = 0.0
    paragraph_id: Optional[int] = 0
    langs: Optional[int] = 0


class LanceModelWikipedia(BaseModel):
    """LanceDB 表的 Pydantic 模型，包含 1024 维向量字段。"""

    id: str
    title: str
    text: str
    url: Optional[str] = ""
    wiki_id: Optional[int] = 0
    views: Optional[float] = 0.0
    paragraph_id: Optional[int] = 0
    langs: Optional[int] = 0
    vector: Vector(1024)


class SearchResult(LanceModel):
    """搜索结果返回模型。"""

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "id": "12345",
                "title": "Machine learning",
                "text": "Machine learning is a subset of artificial intelligence...",
                "url": "https://en.wikipedia.org/wiki/Machine_learning",
                "wiki_id": 233488,
                "views": 15000.0,
                "paragraph_id": 0,
                "langs": 85,
            }
        },
    )

    id: str
    title: str
    text: str
    url: Optional[str] = ""
    wiki_id: Optional[int] = 0
    views: Optional[float] = 0.0
    paragraph_id: Optional[int] = 0
    langs: Optional[int] = 0
