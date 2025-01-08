import asyncio
from src.database.pg_db_ops import DatabaseManager


async def add_product_data():
    await DatabaseManager.add_gadgets_from_csv("../../data/products_consolidated.csv")


async def run_updates():
    await DatabaseManager.load_quiz_data_from_json("quiz_questions.json")
    # Ingest categories first
    await DatabaseManager.add_categories_from_json("categories.json")
    # Then load quiz data
    await DatabaseManager.load_quiz_data_from_json("quiz_questions.json")
