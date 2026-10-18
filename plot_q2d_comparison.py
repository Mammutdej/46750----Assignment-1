"""
Create report-ready plots for Q2(d).

Run after q2d_comparison.py:
    python plot_q2d_comparison.py

Reads results/Q2d_comparison/q2d_summary.csv and q2d_hourly_schedules.csv.
Writes PNG files to figures/ and copies none of the input data.
"""
from pathlib import Path
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "results" / "Q2d_comparison"
FIG_DIR = ROOT / "figures"
LABELS = {
    "Q1_linear_utility": "Q1: Linear utility",
    "Q2b_linear_disutility": "Q2(b): Linear disutility",
    "Q2c_quadratic_disutility": "Q2(c): Quadratic disutility",
}


def save(fig, filename):
    fig.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Created:", path)


def main():
    summary_path = DATA_DIR / "q2d_summary.csv"
    schedule_path = DATA_DIR / "q2d_hourly_schedules.csv"
    if not summary_path.exists() or not schedule_path.exists():
        raise FileNotFoundError(
            f"Missing Q2(d) result files in {DATA_DIR}. "
            "Run python q2d_comparison.py first."
        )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_path)
    hourly = pd.read_csv(schedule_path)

    # 1. Hourly schedule comparison, including reference profile.
    fig, ax = plt.subplots(figsize=(9, 5))
    first = hourly[hourly["model"] == summary.iloc[0]["model"]]
    if "reference_load_kWh" in first and first["reference_load_kWh"].notna().any():
        ax.plot(first["hour"], first["reference_load_kWh"], "k--", linewidth=2,
                label="Reference load")
    for model in LABELS:
        part = hourly[hourly["model"] == model]
        if not part.empty:
            ax.plot(part["hour"], part["optimal_load_kWh"], marker="o",
                    markersize=3, linewidth=1.6, label=LABELS[model])
    ax.set(title="Q2(d): Optimal hourly load under three preference models",
           xlabel="Hour", ylabel="Load (kWh per hour)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save(fig, "q2d_load_profiles.png")

    # 2. Procurement cost and total objective (common DKK units).
    for column, ylabel, filename, title in [
        ("procurement_cost_DKK", "Procurement cost (DKK)", "q2d_procurement_cost.png",
         "Procurement cost by model"),
        ("objective_total_DKK", "Objective value (DKK)", "q2d_objective_values.png",
         "Objective value by model"),
        ("daily_energy_kWh", "Daily energy (kWh)", "q2d_daily_energy.png",
         "Daily energy consumption by model"),
        ("total_absolute_deviation_kWh", "Total absolute deviation (kWh)",
         "q2d_absolute_deviation.png", "Deviation from reference profile"),
        ("hours_at_load_min_or_max", "Hours at minimum or maximum load",
         "q2d_load_bound_hours.png", "Hours at load bounds"),
    ]:
        if column not in summary.columns or summary[column].isna().all():
            print(f"Skipped {filename}: metric {column} is unavailable.")
            continue
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        data = summary.dropna(subset=[column])
        labels = [LABELS.get(m, m) for m in data["model"]]
        ax.bar(labels, data[column])
        ax.set(title=title, ylabel=ylabel)
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y", alpha=0.3)
        save(fig, filename)

    print("\nDone. Figures saved in:", FIG_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
