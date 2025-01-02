from sqlalchemy.ext.asyncio import create_async_engine

from src.configs.vars import VarsConfig

DB_URI = f'postgresql+asyncpg://{VarsConfig.POSTGRES_USER}:{VarsConfig.POSTGRES_PASSWORD}@{VarsConfig.POSTGRES_HOST}:{VarsConfig.POSTGRES_PORT}/{VarsConfig.POSTGRES_DB_NAME}'

# Async engine setup
engine = create_async_engine(DB_URI, echo=False)
