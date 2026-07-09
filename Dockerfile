FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 poppler-utils \
    && pip install --no-cache-dir uv
COPY pyproject.toml ./
RUN uv pip install --system -e ".[default]" || uv pip install --system -e .
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
