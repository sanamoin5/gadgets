import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import insert, select
from pg_models import Base, Gadget, Category, gadget_category_association
from config import engine
import uuid
import datetime


class DatabaseManager:
    """Handles database operations."""

    @staticmethod
    async def add_gadgets_from_csv(file_path: str):
        """Adds gadgets to the database from a CSV file."""
        try:
            # Load the CSV data using Pandas
            data = pd.read_csv(file_path)

            # Process each row in the DataFrame
            async with engine.begin() as conn:
                for _, row in data.iterrows():
                    query = insert(Gadget).values(
                        id=uuid.uuid4(),
                        name=row['name'],
                        description=row.get('description', None),
                        price=float(row['price']) if not pd.isna(row['price']) else None,
                        currency=row.get('currency', 'EUR'),
                        rating=float(row['rating']) if not pd.isna(row['rating']) else None,
                        reviews_count=int(row['reviews_count']) if not pd.isna(row['reviews_count']) else None,
                        image_url=row.get('image_url', None),
                        amazon_choice=str(row.get('amazon_choice', 'False')).lower() == 'true',
                        best_seller=str(row.get('best_seller', 'False')).lower() == 'true',
                        sales_volume=row.get('sales_volume', None),
                        shipping_info=row.get('shipping_info', None),
                        created_at=datetime.datetime.now(datetime.UTC)
                    )
                    await conn.execute(query)
                await conn.commit()
        except (SQLAlchemyError, IOError) as e:
            raise SQLAlchemyError(f"An error occurred while adding gadgets: {e}") from e

    @staticmethod
    async def add_category(category_name: str, description: str = None):
        """Adds a category to the database."""
        try:
            async with engine.begin() as conn:
                query = insert(Category).values(
                    id=uuid.uuid4(),
                    name=category_name,
                    description=description,
                    created_at=datetime.datetime.now(datetime.UTC)
                )
                await conn.execute(query)
                await conn.commit()
        except SQLAlchemyError as e:
            raise SQLAlchemyError(f"An error occurred while adding a category: {e}") from e

    @staticmethod
    async def link_gadget_to_category(gadget_id: uuid.UUID, category_id: uuid.UUID):
        """Links a gadget to a category."""
        try:
            async with engine.begin() as conn:
                query = insert(gadget_category_association).values(
                    gadget_id=gadget_id,
                    category_id=category_id
                )
                await conn.execute(query)
                await conn.commit()
        except SQLAlchemyError as e:
            raise SQLAlchemyError(f"An error occurred while linking gadget to category: {e}") from e

    @staticmethod
    async def get_all_gadgets():
        """Fetches all gadgets."""
        try:
            async with engine.begin() as conn:
                result = await conn.execute(select(Gadget))
                return result.scalars().all()
        except SQLAlchemyError as e:
            raise SQLAlchemyError(f"An error occurred while fetching gadgets: {e}") from e

