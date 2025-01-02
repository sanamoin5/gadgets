from pydantic import BaseModel, Field
from typing import Optional, List


class Question(BaseModel):
    id: int
    question: str
    options: List[str]


class Response(BaseModel):
    question_id: int
    answer: str


class Product(BaseModel):
    id: int
    name: str
    description: str
    price: str
