from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID


class Question(BaseModel):
    id: UUID
    question: str
    options: List[str]


class Response(BaseModel):
    question_id: UUID
    answer: str


class Product(BaseModel):
    id: UUID
    name: str
    description: Optional[str]
    price: Optional[str]


class SBERTInferenceRequest(BaseModel):
    texts: List[str]


class SBERTInferenceResponse(BaseModel):
    embeddings: List[List[float]]


class MetaReviewInferenceRequest(BaseModel):
    review: str
    metadata: str
    price: float


class MetaReviewInferenceResponse(BaseModel):
    score: float
    fused_embedding: list
