from fastapi import Request
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
import time

# Define Prometheus metrics
REQUEST_COUNT = Counter(
    'request_count', 'Total HTTP Requests',
    ['app_name', 'method', 'endpoint', 'http_status']
)

REQUEST_LATENCY = Histogram(
    'request_latency_seconds', 'Request latency',
    ['app_name', 'endpoint']
)


# Middleware to collect metrics
class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        # Update metrics
        REQUEST_COUNT.labels(
            app_name="fastapi_app",
            method=request.method,
            endpoint=request.url.path,
            http_status=response.status_code
        ).inc()

        REQUEST_LATENCY.labels(
            app_name="fastapi_app",
            endpoint=request.url.path
        ).observe(process_time)

        return response
