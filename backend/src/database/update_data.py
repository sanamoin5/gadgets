import asyncio
from src.database.pg_db_ops import DatabaseManager


async def add_product_data():
    await DatabaseManager.add_gadgets_from_csv(
        "../../../recommendation_engine/data/preprocessed/products_consolidated.csv")


async def run_updates():
    # Ingest categories first
    await DatabaseManager.add_categories_from_json("../../../recommendation_engine/data/raw/categories.json")
    # Then load quiz data
    await DatabaseManager.load_quiz_data_from_json("../../../recommendation_engine/data/processed/quiz_data.json")


asyncio.run(run_updates())