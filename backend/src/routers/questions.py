from typing import List
from fastapi import APIRouter
import logging

from src.models.pydantic_schemas import Question

logger = logging.getLogger(__name__)
router = APIRouter()

questions_data = [
    {"id": 1, "question": "What do you primarily use gadgets for?",
     "options": ["Gaming", "Music", "Productivity", "Schoolwork"]},
    {"id": 2, "question": "What's your preferred brand?", "options": ["Apple", "Samsung", "Sony", "Other"]},
    {"id": 3, "question": "What's your budget range?", "options": ["< $100", "$100 - $500", "$500 - $1000", "> $1000"]},
]


@router.get("/questions", response_model=List[Question])
async def fetch_questions():
    """
    Endpoint to fetch quiz questions.
    Returns a list of questions with options.
    """
    return questions_data
