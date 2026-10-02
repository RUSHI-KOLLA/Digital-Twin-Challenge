FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 -r requirements.txt
COPY AGENTS.md README.md ./
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY docker/ ./docker/
COPY src/ ./src/
COPY models/ ./models/
COPY results/ ./results/
COPY configs/ ./configs/
COPY data/synthetic/ ./data/synthetic/
# unified.parquet (2 MB, committed) ships in the image so first boot is
# fully offline. features.parquet is rebuilt in-container from it by the
# app fallbacks (~1 min CPU, no network). data/raw stays out (1.3 GB)
# except tiny bio.csv (static EHR join); the entrypoint downloads the rest
# only if unified.parquet is ever missing.
COPY data/processed/unified.parquet ./data/processed/
COPY data/raw/cgmacros/unzipped/CGMacros/bio.csv ./data/raw/cgmacros/unzipped/CGMacros/bio.csv
RUN chmod +x docker/entrypoint.sh
ENTRYPOINT ["docker/entrypoint.sh"]
EXPOSE 8000 8501
