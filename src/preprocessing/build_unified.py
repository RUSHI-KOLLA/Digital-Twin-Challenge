"""Phase 1: CGMacros -> unified 5-min parquet.

Reads data/raw/cgmacros/unzipped/CGMacros/CGMacros-*/CGMacros-*.csv
into one table: patient_id, timestamp, glucose, hr, steps,
meal_carbs, meal_fat, meal_protein (+ activity_kcal, mets).

Rules (AGENTS.md):
- 5-min grid, glucose gaps <=15 min interpolated, meals/steps summed per bin.
- glucose = Dexcom GL coalesce Libre GL (Dexcom primary, Libre fallback).
- Raw has NO steps column -> steps=0, wearable signal kept as
  activity_kcal (sum) + mets (mean). Documented, no fake steps.
- Meal macros scaled by Amount Consumed % where present.
"""
import glob
import os
import pandas as pd

RAW_GLOB = "data/raw/cgmacros/unzipped/CGMacros/CGMacros-*/CGMacros-*.csv"
OUT = "data/processed/unified.parquet"


def load_one(path: str) -> pd.DataFrame:
    pid = int(os.path.basename(path).split("-")[1].split(".")[0])
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["patient_id"] = pid
    df["timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    # glucose: Dexcom primary, Libre fallback (cols swapped in some files)
    df["glucose"] = df["Dexcom GL"].fillna(df["Libre GL"]) \
        if "Dexcom GL" in df.columns and "Libre GL" in df.columns \
        else df.get("Dexcom GL", df.get("Libre GL"))
    df["hr"] = df.get("HR")
    # steps: present only in a subset of files -> 0 elsewhere (no fake data)
    df["steps"] = df.get("Steps", 0.0)
    df["steps"] = pd.to_numeric(df["steps"], errors="coerce").fillna(0.0)
    if "Calories (Activity)" in df.columns:
        df["activity_kcal"] = pd.to_numeric(
            df["Calories (Activity)"], errors="coerce").fillna(0.0)
    else:
        df["activity_kcal"] = 0.0
    # METs vs Intensity variant -> keep mets, fallback to Intensity
    mets = df.get("METs", df.get("Intensity"))
    df["mets"] = pd.to_numeric(mets, errors="coerce") \
        if mets is not None else float("nan")
    # scale macros by amount consumed % where present
    if "Amount Consumed" in df.columns:
        amt = pd.to_numeric(df["Amount Consumed"],
                            errors="coerce").fillna(100.0) / 100.0
    else:
        amt = 1.0
    for c in ["Carbs", "Protein", "Fat"]:
        df[c] = df[c].fillna(0.0) * amt
    df = df.rename(columns={
        "Carbs": "meal_carbs", "Protein": "meal_protein", "Fat": "meal_fat",
    })
    return df[["patient_id", "timestamp", "glucose", "hr", "steps",
               "meal_carbs", "meal_protein", "meal_fat",
               "activity_kcal", "mets"]]


def resample_patient(g: pd.DataFrame) -> pd.DataFrame:
    g = g.set_index("timestamp").sort_index()
    agg = {
        "glucose": "mean", "hr": "mean", "steps": "sum",
        "meal_carbs": "sum", "meal_protein": "sum", "meal_fat": "sum",
        "activity_kcal": "sum", "mets": "mean",
    }
    out = g.resample("5min").agg({**agg, "patient_id": "first"})
    # interpolate glucose gaps <=15 min (=3 x 5-min bins), interior only
    out["glucose"] = out["glucose"].interpolate(
        method="linear", limit=3, limit_area="inside")
    out = out.dropna(subset=["glucose"])
    out = out.reset_index()
    return out[["patient_id", "timestamp", "glucose", "hr", "steps",
                "meal_carbs", "meal_protein", "meal_fat",
                "activity_kcal", "mets"]]


def main():
    files = sorted(glob.glob(RAW_GLOB))
    assert files, f"no files match {RAW_GLOB}"
    print(f"found {len(files)} subject files")
    parts = [resample_patient(load_one(f)) for f in files]
    u = pd.concat(parts, ignore_index=True).sort_values(
        ["patient_id", "timestamp"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    u.to_parquet(OUT, index=False)
    print(f"saved {OUT} rows={len(u)}")
    print("--- rows per patient ---")
    print(u.groupby("patient_id").size().to_string())
    print("--- current glucose>180 rate (proxy; true event_60 in Phase 2a) ---")
    print(f"{(u['glucose'] > 180).mean() * 100:.2f}%")
    print("--- meals: patients with any carbs>0 ---")
    print(f"{(u.groupby('patient_id')['meal_carbs'].sum() > 0).sum()}/{u['patient_id'].nunique()}")
    # one-patient sanity: first patient, first day range
    pid0 = int(u["patient_id"].iloc[0])
    d0 = u[u["patient_id"] == pid0].iloc[:288]
    print(f"--- sanity patient {pid0} first 288 bins: "
          f"mean={d0['glucose'].mean():.1f} min={d0['glucose'].min():.1f} "
          f"max={d0['glucose'].max():.1f} std={d0['glucose'].std():.1f} ---")


if __name__ == "__main__":
    main()
