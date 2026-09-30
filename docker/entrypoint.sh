#!/bin/bash
# Fresh-clone bootstrap: rebuild processed data if it isn't in the image.
# data/processed/*.parquet is gitignored, so a clone has no features.
# Raw CGMacros is open-access (no credentials): download once, build, serve.
set -e
if [ ! -f data/processed/unified.parquet ]; then
  echo "[entrypoint] no processed data - bootstrapping from open CGMacros..."
  mkdir -p data/raw/cgmacros/unzipped
  if [ ! -f data/raw/cgmacros/CGMacros_dateshifted365.zip ]; then
    aws s3 sync --no-sign-request s3://physionet-open/cgmacros/1.0.0/ data/raw/cgmacros/
  fi
  if [ ! -d data/raw/cgmacros/unzipped/CGMacros ]; then
    python -c "import zipfile; zipfile.ZipFile('data/raw/cgmacros/CGMacros_dateshifted365.zip').extractall('data/raw/cgmacros/unzipped')"
  fi
  python src/preprocessing/build_unified.py
  echo "[entrypoint] bootstrap done."
fi
exec "$@"
