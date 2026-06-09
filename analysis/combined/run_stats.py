"""
Statistical analysis of multimodal stress signals (W1 control vs W2 water deficit).

Analysis 1: OLS — leaf-to-air temperature difference over time (W1 vs W2).
Analysis 2: OLS — hyperspectral index (NDVI, CI_RedEdge, SIPI) over time (W1 vs W2).
Analysis 3: Spearman correlation — daily median turgor vs leaf temperature
            and hyperspectral indices over time, per bed.

All regression models use the fixed-effects structure:
  outcome ~ treatment * days_since_intervention
where treatment contrasts W2 against W1 (reference). Both hyperspectral and thermal data are
aggregated to per-bed daily values (mean and median respectively) before modelling, so each
observation is an independent bed-date point and OLS is appropriate without random effects.

Sensor assignments for turgor:
  W1 (control):       sensors 19–21
  W2 (water deficit): sensors 10–18

Outputs: processed/ sub-directory (OLS tables + correlation CSV).

Developed with assistance from Claude (Anthropic) via Claude Code.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
DATA = ROOT / "data"
ANALYSIS = ROOT / "analysis"
OUT = Path(__file__).resolve().parent
PROC = OUT / "processed"
PROC.mkdir(exist_ok=True)

INTERVENTION = pd.Timestamp("2025-09-03")
ANALYSIS_START = pd.Timestamp("2025-09-01")
ANALYSIS_END = pd.Timestamp("2025-09-14")
HYPERSPECTRAL_INDICES = ["ndvi", "ci_rededge", "sipi"]

W1_SENSORS = list(range(19, 22))   # 19, 20, 21
W2_SENSORS = list(range(10, 19))   # 10–18


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_hyperspectral():
    df = pd.read_excel(DATA / "results_w5_mean_15_sa.xlsx")
    df = df[df["index"].isin(HYPERSPECTRAL_INDICES)].copy()
    df["date"] = pd.to_datetime(df["group"], format="%Y_%m_%d")
    df["days_since_intervention"] = (df["date"] - INTERVENTION).dt.days
    return df[df["bed_id"].isin(["W1", "W2"])].copy()


def load_thermal():
    with open(ANALYSIS / "leaf_air_temp" / "thermal_analysis.json") as f:
        raw = json.load(f)

    rows = []
    for key, records in raw.items():
        bed, plant = key.split("_", 1)
        for r in records:
            row = dict(r)
            row["bed"] = bed
            row["plant"] = plant
            row["plant_id"] = key
            rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y_%m_%d")
    df["hour"] = df["hour"].astype(int)
    df = df[df["hour"] == 15].copy()
    df["days_since_intervention"] = (df["date"] - INTERVENTION).dt.days

    # leaf-to-air diff; W2 averages both air sensors, W1 uses sensor 2
    df["leaf_air_diff"] = df["leaf_temp_c"] - np.where(
        df["bed"] == "W1",
        df["air_temp_sensor2_c"],
        (df["air_temp_sensor1_c"] + df["air_temp_sensor2_c"]) / 2,
    )
    return df[df["bed"].isin(["W1", "W2"])].copy()


def load_turgor():
    df = pd.read_parquet(DATA / "leaf_readings.parquet")
    df["sensor_number"] = df["sensor_number"].astype(int)
    df = df[df["sensor_number"].isin(W1_SENSORS + W2_SENSORS)].copy()
    df["bed"] = np.where(df["sensor_number"].isin(W1_SENSORS), "W1", "W2")
    df["date"] = df["berlin_timestamp"].dt.normalize().dt.tz_localize(None)

    # Each sensor has its own arbitrary range, so raw values are not comparable
    # across sensors. Normalize to [0, 1] per sensor using the full recorded range
    # before restricting to the analysis window.
    stats = df.groupby("sensor_number")["turgic_value"].agg(["min", "max"])
    df = df.join(stats, on="sensor_number")
    rng = (df["max"] - df["min"]).replace(0, np.nan)
    df["turgic_value"] = (df["turgic_value"] - df["min"]) / rng
    df = df.drop(columns=["min", "max"])

    df = df[(df["date"] >= ANALYSIS_START) & (df["date"] <= ANALYSIS_END)].copy()
    return df


# ---------------------------------------------------------------------------
# Analysis 1: OLS — leaf temperature (W1 vs W2)
# ---------------------------------------------------------------------------

def analyse_thermal_ols(df):
    ols_df = (
        df.groupby(["bed", "days_since_intervention"])["leaf_air_diff"]
        .median()
        .reset_index()
        .dropna()
    )
    ols_df["bed"] = ols_df["bed"].astype("category")
    fit = ols(
        "leaf_air_diff ~ C(bed, Treatment('W1')) * days_since_intervention",
        data=ols_df,
    ).fit()
    summary = fit.summary2().tables[1]
    summary.to_csv(PROC / "thermal_ols_summary.csv")
    print(summary)


# ---------------------------------------------------------------------------
# Analysis 2: OLS — hyperspectral indices (W1 vs W2)
# ---------------------------------------------------------------------------

def analyse_hyperspectral_ols(df):
    for idx in HYPERSPECTRAL_INDICES:
        sub = df[df["index"] == idx][
            ["mean", "bed_id", "days_since_intervention"]
        ].dropna().copy()
        sub["bed_id"] = sub["bed_id"].astype("category")
        fit = ols(
            "mean ~ C(bed_id, Treatment('W1')) * days_since_intervention",
            data=sub,
        ).fit()
        summary = fit.summary2().tables[1]
        summary.to_csv(PROC / f"hyperspectral_ols_{idx}.csv")
        print(f"\n{idx}:")
        print(summary)


# ---------------------------------------------------------------------------
# Analysis 3: Spearman correlation — turgor vs thermal and hyperspectral
# ---------------------------------------------------------------------------

def analyse_turgor_correlations(hyp_df, th_df, tur_df):
    # Daily median turgor per bed
    daily_tur = (
        tur_df.groupby(["bed", "date"])["turgic_value"]
        .median()
        .reset_index()
    )
    daily_tur.to_csv(PROC / "turgor_daily_bed.csv", index=False)

    # Daily median leaf-air diff per bed
    daily_th = (
        th_df.groupby(["bed", "date"])["leaf_air_diff"]
        .median()
        .reset_index()
    )

    results = []
    for bed in ["W1", "W2"]:
        tur_s = daily_tur[daily_tur["bed"] == bed].set_index("date")["turgic_value"]
        th_s = daily_th[daily_th["bed"] == bed].set_index("date")["leaf_air_diff"]

        merged = pd.concat([tur_s, th_s], axis=1, sort=True).dropna()
        if len(merged) >= 4:
            rho, p = stats.spearmanr(merged["turgic_value"], merged["leaf_air_diff"])
        else:
            rho, p = np.nan, np.nan
        results.append({
            "bed": bed,
            "predictor": "leaf_air_diff",
            "n": len(merged),
            "spearman_rho": rho,
            "p_value": p,
        })

        for idx in HYPERSPECTRAL_INDICES:
            hyp_s = (
                hyp_df[(hyp_df["index"] == idx) & (hyp_df["bed_id"] == bed)]
                .set_index("date")["mean"]
            )
            merged = pd.concat([tur_s, hyp_s], axis=1, sort=True).dropna()
            if len(merged) >= 4:
                rho, p = stats.spearmanr(merged["turgic_value"], merged["mean"])
            else:
                rho, p = np.nan, np.nan
            results.append({
                "bed": bed,
                "predictor": idx,
                "n": len(merged),
                "spearman_rho": rho,
                "p_value": p,
            })

    res_df = pd.DataFrame(results)
    res_df.to_csv(PROC / "turgor_correlations.csv", index=False)
    return res_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading data...")
    hyp = load_hyperspectral()
    th = load_thermal()
    tur = load_turgor()

    print("\nAnalysis 1: Thermal OLS (W1 vs W2)...")
    analyse_thermal_ols(th)

    print("\nAnalysis 2: Hyperspectral OLS (W1 vs W2)...")
    analyse_hyperspectral_ols(hyp)

    print("\nAnalysis 3: Turgor correlations...")
    res_cor = analyse_turgor_correlations(hyp, th, tur)
    print(res_cor.to_string(index=False))

    print(f"\nProcessed intermediates: {PROC}")
