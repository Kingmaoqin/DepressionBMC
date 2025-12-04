"""Produce paper-ready tables from `final_robust_analysis_results1204`.

The current manuscript needs two focused tables:
1) A "big table" covering only the symptom-score feature set under the
   V2_Exclusive labeling scheme with SMOTE enabled and standard
   StratifiedKFold CV, listing every model configuration aggregated
   across seeds.
2) A separate table for grouped CV results (GroupKFold) under the same
   scheme/feature/SMOTE filter so grouped performance can be optionally
   reported.

This script generates those tables (CSV + LaTeX) in
`paper_ready_results/` for direct manuscript inclusion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

DATA_FILE = Path("final_robust_analysis_results1204")
OUTPUT_DIR = Path("paper_ready_results")


PRIMARY_COLS = [
    "Test_AUC",
    "Test_Acc",
    "Test_F1",
    "Test_Prec",
    "Test_Recall",
    "Test_AUC_CI_Low",
    "Test_AUC_CI_High",
]


def load_results(filepath: Path) -> pd.DataFrame:
    """Load the results CSV with consistent dtypes."""
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    numeric_cols = [c for c in df.columns if c not in {"Seed", "Scheme", "Feature_Set", "SMOTE", "CV_Type", "Model"}]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["SMOTE"] = df["SMOTE"].astype(str).str.lower().map({"true": True, "false": False})
    return df


def filter_slice(df: pd.DataFrame, cv_type: str) -> pd.DataFrame:
    """Filter to V2 + SMOTE + symptom scores for a specific CV type."""
    return df[
        (df["Scheme"] == "V2_Exclusive")
        & (df["Feature_Set"] == "Symptoms_Only")
        & (df["SMOTE"] == True)
        & (df["CV_Type"] == cv_type)
    ].copy()


def aggregate_models(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics across seeds for each model."""
    ordered_cols = [
        "Model",
        "runs",
        "test_auc_mean",
        "test_auc_std",
        "test_auc_ci_low_mean",
        "test_auc_ci_high_mean",
        "test_acc_mean",
        "test_f1_mean",
        "test_prec_mean",
        "test_recall_mean",
        "AUC_with_CI",
    ]

    if df.empty:
        return pd.DataFrame(columns=ordered_cols)

    grouped = (
        df.groupby("Model")
        .agg(
            runs=("Seed", "nunique"),
            **{f"{col.lower()}_mean": (col, "mean") for col in PRIMARY_COLS},
            **{f"{col.lower()}_std": (col, "std") for col in PRIMARY_COLS if not col.endswith("CI_Low") and not col.endswith("CI_High")},
        )
        .reset_index()
    )

    for col in grouped.columns:
        if col.endswith(("_mean", "_std")):
            grouped[col] = grouped[col].round(3)
    grouped["AUC_with_CI"] = grouped.apply(
        lambda r: f"{r['test_auc_mean']:.3f} ± {r['test_auc_std']:.3f} (CI {r['test_auc_ci_low_mean']:.3f}–{r['test_auc_ci_high_mean']:.3f})",
        axis=1,
    )
    return grouped[ordered_cols].sort_values("test_auc_mean", ascending=False).reset_index(drop=True)


def export_table(df: pd.DataFrame, stem: str, caption: str, label: str) -> None:
    """Write CSV + LaTeX versions of a table."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = OUTPUT_DIR / f"{stem}.csv"
    tex_path = OUTPUT_DIR / f"{stem}.tex"
    df.to_csv(csv_path, index=False)
    df.to_latex(
        tex_path,
        index=False,
        float_format="{:.3f}".format,
        caption=caption,
        label=label,
    )
    print(f"[INFO] Saved {csv_path} and {tex_path}")


def build_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create the stratified (main) and grouped tables."""
    strat_df = filter_slice(df, cv_type="StratifiedKFold")
    group_df = filter_slice(df, cv_type="GroupKFold")
    return aggregate_models(strat_df), aggregate_models(group_df)


def main() -> None:
    df = load_results(DATA_FILE)
    strat_table, group_table = build_tables(df)

    export_table(
        strat_table,
        stem="v2_smote_symptoms_stratified",
        caption="All models under V2_Exclusive with symptom scores, SMOTE, StratifiedKFold",
        label="tab:v2_smote_symptoms_stratified",
    )
    export_table(
        group_table,
        stem="v2_smote_symptoms_group",
        caption="GroupKFold results under V2_Exclusive with symptom scores and SMOTE",
        label="tab:v2_smote_symptoms_group",
    )


if __name__ == "__main__":
    main()
