import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.database.db_config import engine
from src.database.pg_db_ops import DatabaseManager
from src.database.pg_models import Base
from src.inference.model_loader import load_sbert_model
from src.middlewares.metrics import MetricsMiddleware
from src.routers import questions, results, all_gadgets
import logging
from src.configs.logging import setup_logging
from starlette.responses import Response

from src.utils.qdrant_loader import load_qdrant_manager

if not os.getenv("TEST_ENVIRONMENT"):
    listener = setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    MetricsMiddleware
)

app.include_router(questions.router, prefix="/api/v1")
app.include_router(results.router, prefix="/api/v1")
app.include_router(all_gadgets.router, prefix="/api/v1")


@app.get("/metrics")
def metrics():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Application starting")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    load_sbert_model()  # JIT SBERT model view1
    await load_qdrant_manager()  # Qdrant client


@app.on_event("shutdown")
async def shutdown_event() -> None:
    listener.stop()
