from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import SQLAlchemyError
import logging

from src.database.pg_db_ops import DatabaseManager
from src.models.pydantic_schemas import Question

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/questions", response_model=List[Question])
async def fetch_questions():
    """
    Endpoint to fetch quiz questions and their options.
    Returns a list of questions with options.
    """
    try:
        questions_data = await DatabaseManager.fetch_questions()
        if not questions_data:
            raise HTTPException(status_code=404, detail="No quiz questions found.")
        return questions_data
    except SQLAlchemyError as e:
        logger.error(f"Error in /questions endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
