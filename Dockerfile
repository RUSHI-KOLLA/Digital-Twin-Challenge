FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY AGENTS.md README.md ./
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY docker/ ./docker/
COPY src/ ./src/
COPY models/ ./models/
COPY results/ ./results/
COPY configs/ ./configs/
COPY data/synthetic/ ./data/synthetic/
# NOTE: no COPY data/processed (gitignored) and no data/raw (1.3 GB):
# the entrypoint rebuilds processed data from open CGMacros on first boot.
RUN chmod +x docker/entrypoint.sh
ENTRYPOINT ["docker/entrypoint.sh"]
EXPOSE 8000 8501
