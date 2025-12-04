"""Produce paper-ready tables from `final_robust_analysis_results1204` without external deps.

Outputs:
1) StratifiedKFold results under V2_Exclusive + Symptoms_Only + SMOTE=True
2) GroupKFold results under the same filter (optional reporting)

Each table aggregates metrics across seeds for every model and is exported as CSV and
basic LaTeX for manuscript inclusion. Standard library only (csv/statistics/pathlib).
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_results(filepath: Path) -> List[Dict[str, object]]:
    """Load CSV rows as dictionaries with typed numeric fields."""
    rows: List[Dict[str, object]] = []
    with filepath.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rec: Dict[str, object] = {k.strip(): v for k, v in raw.items()}
            rec["Seed"] = rec.get("Seed")
            rec["Scheme"] = rec.get("Scheme")
            rec["Feature_Set"] = rec.get("Feature_Set")
            rec["SMOTE"] = _to_bool(rec.get("SMOTE", "False"))
            rec["CV_Type"] = rec.get("CV_Type")
            rec["Model"] = rec.get("Model")
            for col in PRIMARY_COLS + [
                "Test_Acc_CI_Low",
                "Test_Acc_CI_High",
                "Test_F1_CI_Low",
                "Test_F1_CI_High",
                "Test_Prec_CI_Low",
                "Test_Prec_CI_High",
                "Test_Recall_CI_Low",
                "Test_Recall_CI_High",
            ]:
                rec[col] = _to_float(rec.get(col, "nan"))
            rows.append(rec)
    return rows


def filter_slice(rows: Iterable[Dict[str, object]], cv_type: str) -> List[Dict[str, object]]:
    return [
        r
        for r in rows
        if r.get("Scheme") == "V2_Exclusive"
        and r.get("Feature_Set") == "Symptoms_Only"
        and bool(r.get("SMOTE"))
        and r.get("CV_Type") == cv_type
    ]


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_models(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not rows:
        return []

    grouped: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        grouped.setdefault(r["Model"], []).append(r)

    table: List[Dict[str, object]] = []
    for model, entries in grouped.items():
        record: Dict[str, object] = {"Model": model, "runs": len({e.get("Seed") for e in entries})}
        for col in PRIMARY_COLS:
            vals = [_to_float(e.get(col, float("nan"))) for e in entries]
            record[f"{col.lower()}_mean"] = _mean(vals)
            if not col.endswith("CI_Low") and not col.endswith("CI_High"):
                record[f"{col.lower()}_std"] = _std(vals)
        record["AUC_with_CI"] = "{:.3f} ± {:.3f} (CI {:.3f}–{:.3f})".format(
            record.get("test_auc_mean", float("nan")),
            record.get("test_auc_std", float("nan")),
            record.get("test_auc_ci_low_mean", float("nan")),
            record.get("test_auc_ci_high_mean", float("nan")),
        )
        table.append(record)

    order = [
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
    for rec in table:
        for key in list(rec.keys()):
            if key.endswith(("_mean", "_std")) and isinstance(rec[key], float):
                rec[key] = round(rec[key], 3)
    return sorted(table, key=lambda r: r.get("test_auc_mean", 0), reverse=True), order


def export_table(rows: List[Dict[str, object]], columns: List[str], stem: str, caption: str, label: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = OUTPUT_DIR / f"{stem}.csv"
    tex_path = OUTPUT_DIR / f"{stem}.tex"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for r in rows:
            writer.writerow([r.get(col, "") for col in columns])

    with tex_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[!htbp]\n")
        f.write("  \\centering\n")
        f.write(f"  \\caption{{{caption}}}\n")
        f.write(f"  \\label{{{label}}}\n")
        f.write("  \\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|}\n")
        f.write("    \\hline\n")
        f.write("    Model & Runs & AUC & AUC SD & CI Low & CI High & Acc & F1 & Prec & Recall \\\\ \hline\n")
        for r in rows:
            f.write(
                "    {} & {} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\ \hline\n".format(
                    r.get("Model", ""),
                    r.get("runs", ""),
                    r.get("test_auc_mean", float("nan")),
                    r.get("test_auc_std", float("nan")),
                    r.get("test_auc_ci_low_mean", float("nan")),
                    r.get("test_auc_ci_high_mean", float("nan")),
                    r.get("test_acc_mean", float("nan")),
                    r.get("test_f1_mean", float("nan")),
                    r.get("test_prec_mean", float("nan")),
                    r.get("test_recall_mean", float("nan")),
                )
            )
        f.write("  \\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"[INFO] Saved {csv_path} and {tex_path}")


def build_tables(rows: List[Dict[str, object]]) -> Tuple[Tuple[List[Dict[str, object]], List[str]], Tuple[List[Dict[str, object]], List[str]]]:
    strat_rows = filter_slice(rows, cv_type="StratifiedKFold")
    group_rows = filter_slice(rows, cv_type="GroupKFold")
    strat_table, cols = aggregate_models(strat_rows)
    group_table, _ = aggregate_models(group_rows)
    return (strat_table, cols), (group_table, cols)


def main() -> None:
    rows = load_results(DATA_FILE)
    (strat_table, cols), (group_table, _) = build_tables(rows)
    export_table(
        strat_table,
        columns=cols,
        stem="v2_smote_symptoms_stratified",
        caption="V2_Exclusive + Symptoms_Only with SMOTE (StratifiedKFold)",
        label="tab:v2_smote_symptoms_stratified",
    )
    export_table(
        group_table,
        columns=cols,
        stem="v2_smote_symptoms_group",
        caption="V2_Exclusive + Symptoms_Only with SMOTE (GroupKFold)",
        label="tab:v2_smote_symptoms_group",
    )


if __name__ == "__main__":
    main()
