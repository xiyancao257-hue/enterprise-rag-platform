FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY config ./config
COPY data ./data
COPY docs ./docs
COPY tests ./tests

RUN python -m pip install --upgrade pip \
    && python -m pip install uv \
    && uv sync --extra dev --frozen

CMD ["enterprise-rag", "--help"]
