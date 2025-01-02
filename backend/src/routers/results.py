from typing import List

from fastapi import APIRouter
import logging

from src.models.pydantic_schemas import Product, Response

logger = logging.getLogger(__name__)
router = APIRouter()

products_data = [
    {"id": 1, "name": "Xbox Series X", "description": "Gaming console", "price": "$499"},
    {"id": 2, "name": "AirPods Pro", "description": "Wireless earphones", "price": "$249"},
    {"id": 3, "name": "MacBook Pro", "description": "High-performance laptop", "price": "$1999"},
    {"id": 4, "name": "Samsung Galaxy Tab", "description": "Versatile tablet for productivity", "price": "$699"},
]


@router.post("/results", response_model=List[Product])
async def fetch_results(responses: List[Response]):
    """
    Endpoint to process quiz responses and return product recommendations.
    """
    # Dummy logic to generate product recommendations based on responses
    recommendations = []
    for response in responses:
        if response.answer in ["Gaming", "Apple"]:
            recommendations.append(products_data[0])  # Xbox Series X
        elif response.answer in ["Music", "Sony"]:
            recommendations.append(products_data[1])  # AirPods Pro
        elif response.answer in ["Productivity", "Samsung"]:
            recommendations.append(products_data[3])  # Galaxy Tab
        elif response.answer == "> $1000":
            recommendations.append(products_data[2])  # MacBook Pro

    # Avoid duplicates in recommendations
    unique_recommendations = {p["id"]: p for p in recommendations}.values()
    return list(unique_recommendations)
