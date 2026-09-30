FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY AGENTS.md README.md ./
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY src/ ./src/
COPY models/ ./models/
COPY results/ ./results/
COPY configs/ ./configs/
COPY data/processed/ ./data/processed/
COPY data/synthetic/ ./data/synthetic/
EXPOSE 8000 8501
