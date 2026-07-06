FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY data ./data
COPY docs ./docs
COPY tests ./tests

RUN python -m pip install --upgrade pip \
    && python -m pip install -e .

CMD ["enterprise-rag", "--help"]
