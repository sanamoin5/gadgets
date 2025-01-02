FROM python:3.12-slim
LABEL authors="sanamoin"

WORKDIR /app

COPY . /app

RUN apt-get update; \
    apt-get install -y curl; \
    apt-get install -y git; \
    apt-get autoremove; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /usr/share/doc/*;

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt \
    pip cache purge

CMD uvicorn src.main:app --host 0.0.0.0 --port 8000