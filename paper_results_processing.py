"""Generate paper-ready summary tables from final_robust_analysis_results1204.

The script aggregates cross-validation runs by scheme/feature set/model,
computes mean and standard deviation of the primary metrics, and exports
CSV and LaTeX tables suitable for manuscript inclusion.
"""
from pathlib import Path
import pandas as pd

DATA_FILE = Path("final_robust_analysis_results1204")
OUTPUT_DIR = Path("paper_ready_results")


def load_results(filepath: Path) -> pd.DataFrame:
    """Load the final robust analysis results with consistent numeric typing."""
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    numeric_cols = [
        col
        for col in df.columns
        if col
        not in {
            "Seed",
            "Scheme",
            "Feature_Set",
            "SMOTE",
            "CV_Type",
            "Model",
        }
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["SMOTE"] = df["SMOTE"].astype(bool)
    return df


def summarize_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics across seeds for each model configuration."""
    group_cols = ["Scheme", "Feature_Set", "SMOTE", "CV_Type", "Model"]
    summary = (
        df.groupby(group_cols)
        .agg(
            runs=("Seed", "nunique"),
            test_auc_mean=("Test_AUC", "mean"),
            test_auc_std=("Test_AUC", "std"),
            test_acc_mean=("Test_Acc", "mean"),
            test_f1_mean=("Test_F1", "mean"),
            test_prec_mean=("Test_Prec", "mean"),
            test_recall_mean=("Test_Recall", "mean"),
            test_auc_ci_low_mean=("Test_AUC_CI_Low", "mean"),
            test_auc_ci_high_mean=("Test_AUC_CI_High", "mean"),
        )
        .reset_index()
    )

    for col in [
        "test_auc_mean",
        "test_auc_std",
        "test_acc_mean",
        "test_f1_mean",
        "test_prec_mean",
        "test_recall_mean",
        "test_auc_ci_low_mean",
        "test_auc_ci_high_mean",
    ]:
        summary[col] = summary[col].round(3)

    summary["auc_with_ci"] = summary.apply(
        lambda row: f"{row['test_auc_mean']:.3f} ± {row['test_auc_std']:.3f}"
        f" (CI {row['test_auc_ci_low_mean']:.3f}–{row['test_auc_ci_high_mean']:.3f})",
        axis=1,
    )
    return summary


def top_models_by_scheme(summary: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """Select the top-K configurations per labeling scheme and feature set."""
    return (
        summary.sort_values(["Scheme", "Feature_Set", "test_auc_mean"], ascending=[True, True, False])
        .groupby(["Scheme", "Feature_Set"], as_index=False)
        .head(top_k)
    )


def scheme_level_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Compute scheme-level averages to contextualize performance ranges."""
    overview = (
        df.groupby(["Scheme", "Feature_Set"]) 
        .agg(
            runs=("Seed", "nunique"),
            mean_test_auc=("Test_AUC", "mean"),
            median_test_auc=("Test_AUC", "median"),
            mean_test_acc=("Test_Acc", "mean"),
            mean_test_f1=("Test_F1", "mean"),
        )
        .reset_index()
    )
    for col in ["mean_test_auc", "median_test_auc", "mean_test_acc", "mean_test_f1"]:
        overview[col] = overview[col].round(3)
    return overview


def export_tables(summary: pd.DataFrame, top_models: pd.DataFrame, overview: pd.DataFrame) -> None:
    """Write CSV and LaTeX tables for manuscript-ready insertion."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    summary.to_csv(OUTPUT_DIR / "summary_by_model.csv", index=False)
    summary.to_latex(
        OUTPUT_DIR / "summary_by_model.tex",
        index=False,
        float_format="{:.3f}".format,
        caption="Cross-validated performance aggregated across seeds",
        label="tab:summary_by_model",
    )

    top_models.to_csv(OUTPUT_DIR / "top_models_by_scheme.csv", index=False)
    top_models.to_latex(
        OUTPUT_DIR / "top_models_by_scheme.tex",
        index=False,
        float_format="{:.3f}".format,
        caption="Top configurations per labeling scheme",
        label="tab:top_models_by_scheme",
    )

    overview.to_csv(OUTPUT_DIR / "scheme_level_overview.csv", index=False)
    overview.to_latex(
        OUTPUT_DIR / "scheme_level_overview.tex",
        index=False,
        float_format="{:.3f}".format,
        caption="Scheme-level performance overview",
        label="tab:scheme_overview",
    )


def main() -> None:
    df = load_results(DATA_FILE)
    summary = summarize_by_model(df)
    top_models = top_models_by_scheme(summary, top_k=3)
    overview = scheme_level_overview(df)
    export_tables(summary, top_models, overview)
    print(f"[INFO] Saved tables to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
