"""
Create plots for Question 2(b) and 2(c).

Place this file next to main.py and run:
    python plot_q2_results.py

Reads CSVs from results/Q2_analysis/ (or results/Q2_sensitivity/)
and saves figures to results/Q2_analysis/plots/ or the corresponding results folder.
"""
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CANDIDATE_DIRS = [ROOT / "results" / "Q2_analysis", ROOT / "results" / "Q2_sensitivity"]

def find_file(filename):
    for folder in CANDIDATE_DIRS:
        path = folder / filename
        if path.exists():
            return path
    return None

def find_col(df, candidates, description):
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Missing {description}. Available columns: {list(df.columns)}")

def load_sweep(filename):
    path = find_file(filename)
    if path is None:
        raise FileNotFoundError(
            f"Cannot find {filename}. Run q2_analysis.py first. "
            f"Searched: {', '.join(str(p) for p in CANDIDATE_DIRS)}"
        )
    df = pd.read_csv(path)
    coeff = find_col(df, ["coefficient", "c", "c_L", "c_Q"], "coefficient column")
    return path.parent, df.sort_values(coeff), coeff

def plot_metric(outdir, filename, title, ylabel, linear, lx, ly, quadratic, qx, qy):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(linear[lx], linear[ly], marker="o", markersize=3, linewidth=1.5, label="Linear disutility (2b)")
    ax.plot(quadratic[qx], quadratic[qy], marker="o", markersize=3, linewidth=1.5, label="Quadratic disutility (2c)")
    ax.set_title(title)
    ax.set_xlabel("Disutility coefficient (cL for linear; cQ for quadratic)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    base_dir, linear, lx = load_sweep("linear_sweep.csv")
    _, quadratic, qx = load_sweep("quadratic_sweep.csv")
    outdir = base_dir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    # The first name is used by the current script; alternatives support older CSVs.
    metrics = [
        (["procurement_cost_DKK", "procurement_cost"], ["procurement_cost_DKK", "procurement_cost"],
         "Procurement cost (DKK)", "procurement_cost.png"),
        (["disutility_DKK", "disutility"], ["disutility_DKK", "disutility"],
         "Disutility (DKK)", "disutility.png"),
        (["daily_load_kWh", "daily_energy_kWh", "energy_kWh", "daily_energy", "total_load_kWh"],
         ["daily_load_kWh", "daily_energy_kWh", "energy_kWh", "daily_energy", "total_load_kWh"],
         "Daily energy consumption (kWh)", "daily_energy.png"),
        (["total_absolute_deviation_kWh", "abs_deviation_kWh", "total_abs_deviation_kWh"],
         ["total_absolute_deviation_kWh", "abs_deviation_kWh", "total_abs_deviation_kWh"],
         "Total absolute deviation (kWh)", "absolute_deviation.png"),
        (["hours_deviating_from_reference", "deviating_hours", "hours_deviating"],
         ["hours_deviating_from_reference", "deviating_hours", "hours_deviating"],
         "Hours deviating from reference", "deviating_hours.png"),
        (["hours_at_load_min_or_max", "hours_at_bound", "bound_hours"],
         ["hours_at_load_min_or_max", "hours_at_bound", "bound_hours"],
         "Hours at minimum or maximum load", "load_bound_hours.png"),
    ]
    for lin_candidates, quad_candidates, ylabel, filename in metrics:
        try:
            ly = find_col(linear, lin_candidates, ylabel)
            qy = find_col(quadratic, quad_candidates, ylabel)
        except KeyError:
            print(f"Skipping {ylabel}: required column not found.")
            continue
        plot_metric(outdir, filename, f"Sensitivity of {ylabel.lower()}", ylabel,
                    linear, lx, ly, quadratic, qx, qy)

    # Plot the two base-case hourly schedules, matching actual column names.
    schedule_specs = [
        ("Q2b_base_schedule_cL_1.43.csv", "Linear base case, cL = 1.43"),
        ("Q2c_base_schedule_cQ_1.00.csv", "Quadratic base case, cQ = 1.00"),
    ]
    schedules = []
    for filename, label in schedule_specs:
        path = find_file(filename)
        if path is None:
            print(f"Skipping {filename}: file not found.")
            continue
        df = pd.read_csv(path)
        hour = find_col(df, ["hour", "t", "Time", "time"], "hour")
        load = find_col(df, ["optimal_load_kWh", "load", "optimal_load", "L"], "optimal load")
        ref = find_col(df, ["reference_load_kWh", "reference_load", "reference", "ref_load", "l_ref"], "reference load")
        schedules.append((df, hour, load, ref, label))

    if schedules:
        fig, ax = plt.subplots(figsize=(9, 5))
        for df, hour, load, ref, label in schedules:
            ax.plot(df[hour], df[load], marker="o", markersize=3, label=label)
        df, hour, load, ref, _ = schedules[0]
        ax.plot(df[hour], df[ref], "--", linewidth=2, label="Reference load")
        ax.set_title("Optimal hourly load versus reference profile")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Load (kWh in each hour)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "base_hourly_load_profiles.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"\nPlots saved to: {outdir}")
    for path in sorted(outdir.glob("*.png")):
        print(" ", path.name)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
