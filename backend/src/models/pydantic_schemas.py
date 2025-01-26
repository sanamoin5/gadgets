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
