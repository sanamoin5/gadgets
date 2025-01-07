import uuid

from src.database.pg_db_ops import DatabaseManager


async def add_product_data():
    await DatabaseManager.add_gadgets_from_csv("../../data/products_consolidated.csv")


async def other_ops():
    # Add a category
    await DatabaseManager.add_category("Consoles", "Gaming Consoles like Xbox and PlayStation")

    # Link a gadget to a category (example IDs, replace with actual UUIDs)
    await DatabaseManager.link_gadget_to_category(uuid.UUID("gadget_id_here"), uuid.UUID("category_id_here"))

    # Fetch all gadgets
    gadgets = await DatabaseManager.get_all_gadgets()
