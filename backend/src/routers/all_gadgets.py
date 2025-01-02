from fastapi import APIRouter
from typing import List
from src.models.pydantic_schemas import Product

router = APIRouter()

# Sample data for gadgets
gadgets_data = [
    {"id": 1, "name": "Xbox Series X", "description": "Gaming console", "price": "$499", "image": "https://example.com/xbox.jpg"},
    {"id": 2, "name": "AirPods Pro", "description": "Wireless earphones", "price": "$249", "image": "https://example.com/airpods.jpg"},
    {"id": 3, "name": "MacBook Pro", "description": "High-performance laptop", "price": "$1999", "image": "https://example.com/macbook.jpg"},
    {"id": 4, "name": "Samsung Galaxy Tab", "description": "Versatile tablet for productivity", "price": "$699", "image": "https://example.com/galaxy-tab.jpg"},
]

@router.get("/gadgets", response_model=List[Product])
async def get_all_gadgets():
    """
    Endpoint to fetch all available gadgets.
    """
    return gadgets_data
