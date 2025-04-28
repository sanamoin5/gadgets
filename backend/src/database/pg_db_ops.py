import asyncio
import json
import logging

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import insert, select
from .pg_models import Gadget, Category, gadget_category_association, QuizQuestion, QuizOption
from .db_config import engine
import uuid
import datetime

logger = logging.getLogger(__name__)


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
        """Adds a single category to the database."""
        try:
            async with engine.begin() as conn:
                # Check if the category already exists
                result = await conn.execute(
                    select(Category).where(Category.name == category_name)
                )
                existing_category = result.scalars().first()

                if not existing_category:
                    # Insert new category
                    query = insert(Category).values(
                        id=uuid.uuid4(),
                        name=category_name,
                        description=description,
                        created_at=datetime.datetime.utcnow()
                    )
                    await conn.execute(query)
                    logger.info(f"Added category: {category_name}")
                else:
                    logger.info(f"Category '{category_name}' already exists.")

                await conn.commit()
        except SQLAlchemyError as e:
            raise SQLAlchemyError(f"An error occurred while adding a category: {e}") from e

    @staticmethod
    async def add_categories_from_json(json_file_path: str):
        """Adds multiple categories from a JSON file to the database."""
        try:
            # Load categories from JSON
            with open(json_file_path, 'r') as file:
                data = json.load(file)
                categories = data.get("categories", [])

            if not categories:
                logger.info("No categories found in the JSON file.")
                return

            async with engine.begin() as conn:
                for category_name in categories:
                    # Check if the category already exists
                    result = await conn.execute(
                        select(Category).where(Category.name == category_name)
                    )
                    existing_category = result.scalars().first()

                    if not existing_category:
                        # Insert new category
                        query = insert(Category).values(
                            id=uuid.uuid4(),
                            name=category_name,
                            description=None,  # Add descriptions if available
                            created_at=datetime.datetime.utcnow()
                        )
                        await conn.execute(query)
                        logger.info(f"Added category: {category_name}")
                    else:
                        logger.info(f"Category '{category_name}' already exists.")

            await conn.commit()
            logger.info(f"Ingested {len(categories)} categories into the database.")

        except (SQLAlchemyError, IOError) as e:
            raise SQLAlchemyError(f"An error occurred while adding categories: {e}") from e

    @staticmethod
    async def load_quiz_data_from_json(file_path: str):
        """
        Loads quiz questions and their options from a JSON file into the database.
        Also links categories and interests to each quiz question.
        """
        try:
            # Load quiz questions from JSON
            with open(file_path, 'r') as json_file:
                questions = json.load(json_file)

            async with engine.begin() as conn:
                for question in questions:
                    # Extract question details
                    question_text = question.get('question')
                    question_type = question.get('question_type', 'sub')  # Default to 'sub' if not specified
                    categories = question.get('categories', [])
                    interests = question.get('interests', [])
                    options = question.get('options', [])

                    if not question_text or not options:
                        logger.info("Skipping invalid question entry.")
                        continue

                    # Ensure all categories exist
                    for category_name in categories:
                        result = await conn.execute(
                            select(Category).where(Category.name == category_name)
                        )
                        existing_category = result.scalars().first()
                        if not existing_category:
                            # Automatically add missing categories
                            query = insert(Category).values(
                                id=uuid.uuid4(),
                                name=category_name,
                                description=None,
                                created_at=datetime.datetime.utcnow()
                            )
                            await conn.execute(query)
                            logger.info(f"Added missing category: {category_name}")

                    # Create a new QuizQuestion entry
                    quiz_question_id = uuid.uuid4()
                    quiz_question = QuizQuestion(
                        id=quiz_question_id,
                        question=question_text,
                        question_type=question_type,
                        categories=', '.join(categories),  # Store as comma-separated string
                        interests=', '.join(interests),  # Store as comma-separated string
                        created_at=datetime.datetime.utcnow()
                    )
                    query = insert(QuizQuestion).values(
                        id=quiz_question.id,
                        question=quiz_question.question,
                        question_type=quiz_question.question_type,
                        categories=quiz_question.categories,
                        interests=quiz_question.interests,
                        created_at=quiz_question.created_at
                    )
                    await conn.execute(query)

                    # Add all associated QuizOptions
                    for option_text in options:
                        option_id = uuid.uuid4()
                        quiz_option = QuizOption(
                            id=option_id,
                            question_id=quiz_question_id,
                            option_text=option_text
                        )
                        option_query = insert(QuizOption).values(
                            id=quiz_option.id,
                            question_id=quiz_option.question_id,
                            option_text=quiz_option.option_text
                        )
                        await conn.execute(option_query)

                # Commit all changes
                await conn.commit()
            logger.info(f"Ingested {len(questions)} quiz questions into the database.")

        except (SQLAlchemyError, IOError, KeyError) as e:
            raise SQLAlchemyError(f"An error occurred while loading quiz data: {e}") from e

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
                logger.info(f"Linked gadget {gadget_id} to category {category_id}")
        except SQLAlchemyError as e:
            raise SQLAlchemyError(f"An error occurred while linking gadget to category: {e}") from e

    @staticmethod
    async def get_all_gadgets():
        """Fetches all gadgets."""
        try:
            async with engine.begin() as conn:
                result = await conn.execute(select(Gadget))
                gadgets = result.scalars().all()
                logger.info(f"Fetched {len(gadgets)} gadgets from the database.")
                return gadgets
        except SQLAlchemyError as e:
            raise SQLAlchemyError(f"An error occurred while fetching gadgets: {e}") from e

    @staticmethod
    async def fetch_questions():
        """
        Fetches quiz questions and their options from the database.
        Returns a list of questions with their options.
        """
        try:
            async with engine.begin() as conn:
                questions_query = await conn.execute(select(QuizQuestion))
                questions = questions_query.scalars().all()

                if not questions:
                    logger.info("No quiz questions found in the database.")
                    return []

                questions_data = []
                for question in questions:
                    options_query = await conn.execute(
                        select(QuizOption.option_text).where(QuizOption.question_id == question)
                    )
                    options = [option[0] for option in options_query.fetchall()]

                    questions_data.append({
                        "id": question,
                        "question": question.question,
                        "options": options,
                    })

                logger.info(f"Fetched {len(questions_data)} quiz questions from the database.")
                return questions_data

        except SQLAlchemyError as e:
            raise SQLAlchemyError(f"An error occurred while fetching quiz questions: {e}") from e


async def run_updates():
    # Ingest categories first
    await DatabaseManager.add_categories_from_json("../../../recommendation_engine/data/raw/categories.json")
    # Then load quiz data
    await DatabaseManager.load_quiz_data_from_json("../../../recommendation_engine/data/processed/quiz_data.json")

# asyncio.run(run_updates())
