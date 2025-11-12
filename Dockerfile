FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
COPY uv.lock .
RUN uv sync --locked

# Copy application code
COPY src/ /app/src/
COPY templates/ /app/templates/

# Copy default configurations
COPY config/ /app/config/

# Create necessary directories
RUN mkdir -p /app/user_configs /app/data /app/output_datasets /app/logs
RUN chmod -R 755 /app/user_configs /app/data /app/output_datasets /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV SERVICE_HOST=0.0.0.0
ENV SERVICE_PORT=8000

# Run application
CMD ["uv", "run", "python", "src/main.py", "--host", "0.0.0.0", "--port", "8000"]