"""HTTP request contracts. Shared domain models remain in src.schemas."""
from pydantic import BaseModel


class DemoRequest(BaseModel):
    example_id: str
    use_milvus: bool = False
