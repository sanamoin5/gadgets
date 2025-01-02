import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.middlewares.metrics import MetricsMiddleware
from src.routers import questions, results
import logging
from src.configs.logging import setup_logging
from starlette.responses import Response

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
    MetricsMiddleware,
    exempt_paths=["/metrics"]
)

app.include_router(questions.router, prefix="/api/v1")
app.include_router(results.router, prefix="/api/v1")


@app.get("/metrics")
def metrics():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Application starting")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    listener.stop()
