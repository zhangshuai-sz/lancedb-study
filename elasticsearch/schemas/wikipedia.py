"""Elasticsearch 端的 Wikipedia 文档数据模型。"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, model_validator


class Wikipedia(BaseModel):
    """Cohere/wikipedia-2023-11-embed-multilingual-v3 数据集的文档模型。"""

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

    @model_validator(mode="before")
    def _create_es_id(cls, values):
        """为 ES 设置 _id 主键字段。"""
        values["_id"] = values.get("id", "")
        return values


class SearchResult(BaseModel):
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
