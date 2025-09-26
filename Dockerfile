FROM python:3.10-slim

WORKDIR /app

# Install OS deps and create data dir in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/data

# Copy requirements and install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, artifacts, plots, and dataset
COPY src/ ./src/
COPY model_utils.py ./
COPY artifacts/ ./artifacts/
COPY plots/ ./plots/
COPY heart_disease_dataset.csv ./

EXPOSE 8002

# Default command: start API
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8002}"]