# GlucoTwin — T2D Digital Twin (PoC)

> We don't predict diabetes. We predict what is about to happen to this patient.

Doctor-facing twin that warns ~60 min in advance when glucose will exceed 180 mg/dL after a meal — and simulates what would prevent it.

See `AGENTS.md` for locked design + build queue (single source of truth).

## Quickstart (Phase 0)
- `python3 -m venv .venv && source .venv/bin/activate`
- `pip install -r requirements.txt`
- CGMacros in `data/raw/cgmacros/` via `aws s3 sync --no-sign-request s3://physionet-open/cgmacros/1.0.0/ data/raw/cgmacros/`
- Phase 1: `python src/preprocessing/build_unified.py`

Status: Phase 0 in progress.
