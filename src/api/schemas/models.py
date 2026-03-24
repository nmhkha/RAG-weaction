from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class SourceMetadata(BaseModel):
    source: str
    chunk_id: int
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceMetadata]