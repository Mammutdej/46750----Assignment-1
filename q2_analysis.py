"""Numerical analysis for Assignment 1, Q2(b) and Q2(c).

This script is written against the actual starter-code API in src/data_loader.py.
Place it in the repository root (the folder containing main.py), then run:
    python q2_analysis.py

Requires the packages in requirements.txt and a working Gurobi licence.
Outputs are saved in results/Q2_analysis/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from src.data_loader import load_question

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "Q2_analysis"
TOL = 1e-5


def solve_q2(kind: str, coefficient: float):
    """Solve one of the provided Q2 cases using its actual starter-code data."""
    if kind == "linear":
        case = "Q2_linear"
    elif kind == "quadratic":
        case = "Q2_quadratic"
    else:
        raise ValueError("kind must be 'linear' or 'quadratic'")

    d = load_question(case)
    if d.reference_load is None:
        raise ValueError(f"{case} has no reference_load in its loaded data")

    # Verify the expected starter-code fields and keep all physical/market data unchanged.
    if kind == "linear" and d.linear_disutility is None:
        raise ValueError("The Q2_linear case has no linear_disutility coefficient")
    if kind == "quadratic" and d.quadratic_disutility is None:
        raise ValueError("The Q2_quadratic case has no quadratic_disutility coefficient")

    m = gp.Model(f"Q2_{kind}_{coefficient:g}")
    m.Params.OutputFlag = 0
    T = range(d.n_hours)

    # Hourly variables; the load bounds are those exposed by InputData.
    load = m.addVars(T, lb=d.load_min_kWh, ub=d.load_max_kWh, name="load")
    pv = m.addVars(T, lb=0.0, name="pv")
    imp = m.addVars(T, lb=0.0, name="import")
    exp = m.addVars(T, lb=0.0, name="export")

    m.addConstrs((pv[t] <= float(d.pv_available[t]) for t in T), name="pv_available")
    # Same sign convention as the assignment: load + export - import - PV = 0.
    m.addConstrs((load[t] + exp[t] - imp[t] - pv[t] == 0 for t in T), name="energy_balance")
    if d.max_import_kW is not None:
        m.addConstrs((imp[t] <= d.max_import_kW for t in T), name="import_limit")
    if d.max_export_kW is not None:
        m.addConstrs((exp[t] <= d.max_export_kW for t in T), name="export_limit")

    # Assignment cost convention: import price = spot + import tariff;
    # export revenue = spot - export tariff; PV incurs its marginal production cost.
    procurement = gp.quicksum(
        (float(d.energy_price[t]) + d.import_tariff) * imp[t]
        - (float(d.energy_price[t]) - d.export_tariff) * exp[t]
        + d.pv_marginal_cost * pv[t]
        for t in T
    )

    if kind == "linear":
        # Absolute deviation LP formulation using non-negative auxiliary variables.
        dev_pos = m.addVars(T, lb=0.0, name="deviation_positive")
        dev_neg = m.addVars(T, lb=0.0, name="deviation_negative")
        m.addConstrs(
            (load[t] - float(d.reference_load[t]) == dev_pos[t] - dev_neg[t] for t in T),
            name="deviation_link",
        )
        disutility = gp.quicksum(float(coefficient) * (dev_pos[t] + dev_neg[t]) for t in T)
    else:
        disutility = gp.quicksum(
            float(coefficient) * (load[t] - float(d.reference_load[t])) ** 2
            for t in T
        )

    m.setObjective(procurement + disutility, GRB.MINIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"{kind}, coefficient={coefficient}: Gurobi status {m.Status}")

    L = np.array([load[t].X for t in T], dtype=float)
    PV = np.array([pv[t].X for t in T], dtype=float)
    I = np.array([imp[t].X for t in T], dtype=float)
    E = np.array([exp[t].X for t in T], dtype=float)
    R = np.asarray(d.reference_load, dtype=float)
    absdev = np.abs(L - R)
    procurement_value = float(procurement.getValue())
    if kind == "linear":
        disutility_value = float(coefficient) * float(absdev.sum())
    else:
        disutility_value = float(coefficient) * float(np.square(L - R).sum())

    hourly = pd.DataFrame({
        "hour": list(T),
        "reference_load_kWh": R,
        "optimal_load_kWh": L,
        "pv_used_kWh": PV,
        "pv_available_kWh": np.asarray(d.pv_available, dtype=float),
        "import_kWh": I,
        "export_kWh": E,
        "energy_price_DKK_per_kWh": np.asarray(d.energy_price, dtype=float),
        "effective_import_price_DKK_per_kWh": np.asarray(d.energy_price, dtype=float) + d.import_tariff,
        "effective_export_price_DKK_per_kWh": np.asarray(d.energy_price, dtype=float) - d.export_tariff,
        "absolute_deviation_kWh": absdev,
        "load_minus_reference_kWh": L - R,
    })
    summary = {
        "model": kind,
        "coefficient": float(coefficient),
        "objective_total_DKK": float(m.ObjVal),
        "procurement_cost_DKK": procurement_value,
        "disutility_DKK": disutility_value,
        "daily_energy_kWh": float(L.sum()),
        "total_absolute_deviation_kWh": float(absdev.sum()),
        "maximum_load_above_reference_kWh": float(np.maximum(L - R, 0).max()),
        "hours_deviating_from_reference": int((absdev > TOL).sum()),
        "hours_at_load_min_or_max": int(
            ((np.abs(L - d.load_min_kWh) <= TOL) |
             (np.abs(L - d.load_max_kWh) <= TOL)).sum()
        ),
        "daily_import_kWh": float(I.sum()),
        "daily_export_kWh": float(E.sum()),
        "daily_pv_used_kWh": float(PV.sum()),
    }
    return summary, hourly


def linear_sweep_values():
    """Include the base value, a broad grid, and +/-0.005 around likely LP breakpoints."""
    d = load_question("Q2_linear")
    base = float(d.linear_disutility)
    # Relevant thresholds: PV marginal cost; effective export prices where export can
    # compete with self-consumption; effective import prices where reference exceeds PV.
    thresholds = [float(d.pv_marginal_cost)]
    effective_import = np.asarray(d.energy_price, dtype=float) + d.import_tariff
    effective_export = np.asarray(d.energy_price, dtype=float) - d.export_tariff
    ref = np.asarray(d.reference_load, dtype=float)
    pv_avail = np.asarray(d.pv_available, dtype=float)
    thresholds.extend(effective_import[ref > pv_avail + TOL].tolist())
    thresholds.extend(effective_export[effective_export > d.pv_marginal_cost + TOL].tolist())
    values = set(np.round(np.arange(0.1, 5.0001, 0.1), 3).tolist())
    values.add(round(base, 3))
    for threshold in thresholds:
        for delta in (-0.005, 0.0, 0.005):
            value = round(float(threshold) + delta, 3)
            if 0.0 < value <= 5.0:
                values.add(value)
    return sorted(values)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    linear_data = load_question("Q2_linear")
    quadratic_data = load_question("Q2_quadratic")
    cL_base = float(linear_data.linear_disutility)
    cQ_base = float(quadratic_data.quadratic_disutility)

    configs = [
        ("linear", linear_sweep_values(), cL_base),
        ("quadratic", sorted(set(np.round(np.r_[np.arange(0.1, 5.0001, 0.1), cQ_base], 3).tolist())), cQ_base),
    ]
    all_summaries = {}
    base_hourly = {}

    for kind, coefficients, base_coefficient in configs:
        summaries = []
        for c in coefficients:
            summary, hourly = solve_q2(kind, float(c))
            summaries.append(summary)
            if abs(float(c) - base_coefficient) < 1e-9:
                base_hourly[kind] = hourly
            print(
                f"{kind:9s} c={c:7.3f} | procurement={summary['procurement_cost_DKK']:8.3f} DKK"
                f" | disutility={summary['disutility_DKK']:8.3f} DKK"
                f" | energy={summary['daily_energy_kWh']:7.3f} kWh"
                f" | abs deviation={summary['total_absolute_deviation_kWh']:7.3f} kWh"
                f" | deviating hours={summary['hours_deviating_from_reference']:2d}"
            )
        table = pd.DataFrame(summaries).sort_values("coefficient")
        table.to_csv(OUT / f"{kind}_sweep.csv", index=False)
        all_summaries[kind] = table

    base_hourly["linear"].to_csv(OUT / f"Q2b_base_schedule_cL_{cL_base:g}.csv", index=False)
    base_hourly["quadratic"].to_csv(OUT / f"Q2c_base_schedule_cQ_{cQ_base:g}.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(base_hourly["linear"]["hour"], base_hourly["linear"]["reference_load_kWh"], "k--", label="Reference")
    ax.plot(base_hourly["linear"]["hour"], base_hourly["linear"]["optimal_load_kWh"], "o-", label=f"Linear, cL={cL_base:g}")
    ax.plot(base_hourly["quadratic"]["hour"], base_hourly["quadratic"]["optimal_load_kWh"], "s-", label=f"Quadratic, cQ={cQ_base:g}")
    ax.set(xlabel="Hour", ylabel="Load (kWh in hour)", title="Q2 base-case schedules")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "base_schedules.png", dpi=160); plt.close(fig)

    metrics = [
        ("procurement_cost_DKK", "Procurement cost (DKK)", "procurement_cost"),
        ("disutility_DKK", "Disutility (DKK)", "disutility"),
        ("daily_energy_kWh", "Daily energy (kWh)", "daily_energy"),
        ("total_absolute_deviation_kWh", "Total absolute deviation (kWh)", "absolute_deviation"),
        ("hours_at_load_min_or_max", "Hours at minimum or maximum load", "load_bound_hours"),
    ]
    for column, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for kind, label in (("linear", "Linear cL"), ("quadratic", "Quadratic cQ")):
            table = all_summaries[kind]
            ax.plot(table["coefficient"], table[column], label=label)
        ax.set(xlabel="Disutility coefficient (units differ between models)", ylabel=ylabel, title=ylabel)
        ax.legend(); fig.tight_layout(); fig.savefig(OUT / f"sensitivity_{filename}.png", dpi=160); plt.close(fig)

    print(f"\nFinished. Results saved in: {OUT}")
    print("Note: the script could be syntax-checked here, but a Gurobi solve requires Gurobi installed and licensed on your machine.")

if __name__ == "__main__":
    main()
