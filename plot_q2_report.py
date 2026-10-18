"""
Create the report figures needed for Q2(c), plus one focused comparison with Q2(b).

Run from the folder containing q2_analysis.py:
    python q2_analysis.py
    python plot_q2_report.py

The analysis script must run successfully first and create:
    results/Q2_analysis/quadratic_sweep.csv
    results/Q2_analysis/linear_sweep.csv
    results/Q2_analysis/Q2c_base_schedule_cQ_<value>.csv
    results/Q2_analysis/Q2b_base_schedule_cL_<value>.csv

Figures are saved to results/Q2_analysis/report_figures/.
This script only reads existing CSVs; it does not solve the optimization model.
"""
from pathlib import Path
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULT_DIRS = [
    ROOT / "results" / "Q2_analysis",
    ROOT / "results" / "Q2_sensitivity",
]


def find_file(pattern, required=True):
    for folder in RESULT_DIRS:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]
    if required:
        searched = "\n".join(f"  {folder / pattern}" for folder in RESULT_DIRS)
        raise FileNotFoundError(
            f"Could not find input matching {pattern!r}.\nSearched:\n{searched}\n"
            "Run q2_analysis.py successfully first."
        )
    return None


def col(df, options, purpose, required=True):
    for name in options:
        if name in df.columns:
            return name
    if required:
        raise KeyError(
            f"Could not find the {purpose} column. Available columns: {list(df.columns)}"
        )
    return None


def numeric(df, column):
    return pd.to_numeric(df[column], errors="coerce")


def save(fig, outdir, filename):
    fig.tight_layout()
    path = outdir / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Created: {path}")


def sweep_plot(df, xcol, ycol, title, xlabel, ylabel, outdir, filename):
    if ycol not in df.columns:
        print(f"Skipped {filename}: '{ycol}' is not in the sweep CSV.")
        return
    x = numeric(df, xcol)
    y = numeric(df, ycol)
    mask = x.notna() & y.notna()
    if not mask.any():
        print(f"Skipped {filename}: no numeric data for '{ycol}'.")
        return
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(x[mask], y[mask], marker="o", markersize=2.5, linewidth=1.7)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(True, alpha=0.3)
    save(fig, outdir, filename)


def load_base_schedule(pattern):
    path = find_file(pattern, required=False)
    if path is None:
        return None, None
    df = pd.read_csv(path)
    h = col(df, ["hour", "t", "Time", "time"], "hour", required=False)
    load = col(df, ["optimal_load_kWh", "optimal_load", "load", "L"], "optimal load", required=False)
    ref = col(df, ["reference_load_kWh", "reference_load", "reference", "ref_load"], "reference load", required=False)
    if h is None or load is None:
        print(f"Skipped schedule {path.name}: missing hour or optimal-load columns.")
        return None, None
    return (path, df, h, load, ref)


def plot_q2c_base(outdir):
    item = load_base_schedule("Q2c_base_schedule_*.csv")
    if item[0] is None:
        print("Skipped Q2(c) base schedule: no usable Q2c schedule CSV found.")
        return
    path, df, h, load, ref = item
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(df[h], df[load], "o-", markersize=3.5, linewidth=1.8, label="Optimal load (Q2c)")
    if ref:
        ax.plot(df[h], df[ref], "k--", linewidth=2, label="Reference load")
    ax.set(title="Q2(c): Base-case optimal load profile",
           xlabel="Hour", ylabel="Load (kWh per hour)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save(fig, outdir, "q2c_base_case_load_profile.png")


def plot_q2b_q2c_comparison(outdir):
    linear_item = load_base_schedule("Q2b_base_schedule_*.csv")
    quad_item = load_base_schedule("Q2c_base_schedule_*.csv")
    if linear_item[0] is None or quad_item[0] is None:
        print("Skipped Q2(b)-Q2(c) comparison: both base schedule CSVs are required.")
        return

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ref_drawn = False
    for item, label, marker in [
        (linear_item, "Linear disutility (Q2b)", "o"),
        (quad_item, "Quadratic disutility (Q2c)", "s"),
    ]:
        _, df, h, load, ref = item
        ax.plot(df[h], df[load], marker + "-", markersize=3, linewidth=1.7, label=label)
        if ref and not ref_drawn:
            ax.plot(df[h], df[ref], "k--", linewidth=2, label="Reference load")
            ref_drawn = True
    ax.set(title="Base-case schedules: Q2(b) versus Q2(c)",
           xlabel="Hour", ylabel="Load (kWh per hour)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save(fig, outdir, "q2b_q2c_base_schedule_comparison.png")


def main():
    quadratic_path = find_file("quadratic_sweep.csv")
    quadratic = pd.read_csv(quadratic_path)
    qx = col(quadratic, ["coefficient", "cQ", "c_Q", "c"], "quadratic coefficient")
    quadratic[qx] = numeric(quadratic, qx)
    quadratic = quadratic.dropna(subset=[qx]).sort_values(qx)

    # Put the figures beside the found analysis output, not in the current working directory.
    outdir = quadratic_path.parent / "report_figures"
    outdir.mkdir(parents=True, exist_ok=True)

    # Main Q2(c) sensitivity plots. Keep them separate because units differ.
    plots = [
        ("procurement_cost_DKK", "Q2(c): Procurement cost versus quadratic coefficient",
         "Quadratic coefficient $c_Q$", "Procurement cost (DKK)", "q2c_procurement_cost_sweep.png"),
        ("disutility_DKK", "Q2(c): Quadratic disutility versus coefficient",
         "Quadratic coefficient $c_Q$", "Quadratic disutility (DKK)", "q2c_disutility_sweep.png"),
        ("objective_total_DKK", "Q2(c): Total objective versus quadratic coefficient",
         "Quadratic coefficient $c_Q$", "Total objective (DKK)", "q2c_total_objective_sweep.png"),
        ("total_absolute_deviation_kWh", "Q2(c): Total absolute deviation versus coefficient",
         "Quadratic coefficient $c_Q$", "Total absolute deviation (kWh)", "q2c_absolute_deviation_sweep.png"),
    ]
    for metric, title, xlabel, ylabel, filename in plots:
        sweep_plot(quadratic, qx, metric, title, xlabel, ylabel, outdir, filename)

    # One comparison figure is useful; do not duplicate every sweep plot for Q2(b).
    plot_q2c_base(outdir)
    plot_q2b_q2c_comparison(outdir)

    # Print a concise list of generated figures and the base coefficient row for checking.
    print("\nFigures created:")
    for path in sorted(outdir.glob("*.png")):
        print(f"  {path.name}")

    base_coeff = None
    # The configured base cQ can be identified from the hourly schedule filename.
    schedule_path = find_file("Q2c_base_schedule_*.csv", required=False)
    if schedule_path:
        import re
        match = re.search(r"cQ_([-+0-9.eE]+)", schedule_path.stem)
        if match:
            try:
                base_coeff = float(match.group(1))
            except ValueError:
                pass
    if base_coeff is not None:
        row = quadratic.loc[(quadratic[qx] - base_coeff).abs() < 1e-7]
        if not row.empty:
            print(f"\nQ2(c) base-case sweep row (cQ={base_coeff:g}):")
            for key in ["procurement_cost_DKK", "disutility_DKK", "objective_total_DKK",
                        "daily_energy_kWh", "total_absolute_deviation_kWh",
                        "hours_deviating_from_reference", "hours_at_load_min_or_max"]:
                if key in row.columns:
                    print(f"  {key}: {row.iloc[0][key]}")
    print(f"\nDone. Output folder: {outdir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
