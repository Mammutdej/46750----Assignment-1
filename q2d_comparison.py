"""
Q2(d): compare Q1 linear utility, Q2(b) linear disutility, and Q2(c) quadratic disutility.

Run from the repository root (the folder containing src/):
    python q2d_comparison.py

Requires Gurobi, pandas, numpy, and the provided src.data_loader module.
Outputs: results/Q2d_comparison/
"""
from pathlib import Path
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from src.data_loader import load_question

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "Q2d_comparison"
TOL = 1e-5

# Q1's base-case marginal utility (DKK/kWh), as used in the report.
# Change this if your Q1 base case uses a different u_L.
U_L_Q1 = 1.43


def solve_model(model_kind, data, coefficient):
    """Solve one model with the same market/physical data."""
    m = gp.Model(f"Q2d_{model_kind}")
    m.Params.OutputFlag = 0
    T = range(data.n_hours)

    load = m.addVars(T, lb=data.load_min_kWh, ub=data.load_max_kWh, name="load")
    pv = m.addVars(T, lb=0.0, name="pv")
    imp = m.addVars(T, lb=0.0, name="import")
    exp = m.addVars(T, lb=0.0, name="export")

    m.addConstrs((pv[t] <= float(data.pv_available[t]) for t in T), name="pv_available")
    m.addConstrs((load[t] + exp[t] - imp[t] - pv[t] == 0 for t in T), name="balance")
    if data.max_import_kW is not None:
        m.addConstrs((imp[t] <= data.max_import_kW for t in T), name="import_limit")
    if data.max_export_kW is not None:
        m.addConstrs((exp[t] <= data.max_export_kW for t in T), name="export_limit")

    procurement = gp.quicksum(
        (float(data.energy_price[t]) + data.import_tariff) * imp[t]
        - (float(data.energy_price[t]) - data.export_tariff) * exp[t]
        + data.pv_marginal_cost * pv[t]
        for t in T
    )

    if model_kind == "Q1_linear_utility":
        disutility = -float(coefficient) * gp.quicksum(load[t] for t in T)
    elif model_kind == "Q2b_linear_disutility":
        dev_pos = m.addVars(T, lb=0.0, name="dev_pos")
        dev_neg = m.addVars(T, lb=0.0, name="dev_neg")
        m.addConstrs(
            (load[t] - float(data.reference_load[t]) == dev_pos[t] - dev_neg[t]
             for t in T), name="deviation_link"
        )
        disutility = float(coefficient) * gp.quicksum(dev_pos[t] + dev_neg[t] for t in T)
    elif model_kind == "Q2c_quadratic_disutility":
        disutility = gp.quicksum(
            float(coefficient) * (load[t] - float(data.reference_load[t])) ** 2
            for t in T
        )
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")

    m.setObjective(procurement + disutility, GRB.MINIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"{model_kind} failed; Gurobi status={m.Status}")

    L = np.array([load[t].X for t in T])
    PV = np.array([pv[t].X for t in T])
    I = np.array([imp[t].X for t in T])
    E = np.array([exp[t].X for t in T])
    R = (np.asarray(data.reference_load, dtype=float)
         if data.reference_load is not None else np.full(data.n_hours, np.nan))
    absdev = np.abs(L - R) if np.isfinite(R).all() else np.full(data.n_hours, np.nan)
    procurement_val = float(procurement.getValue())
    objective_val = float(m.ObjVal)

    if model_kind == "Q1_linear_utility":
        preference_term = -float(coefficient) * float(L.sum())
        preference_label = "utility_term_DKK"
    elif model_kind == "Q2b_linear_disutility":
        preference_term = float(coefficient) * float(absdev.sum())
        preference_label = "disutility_DKK"
    else:
        preference_term = float(coefficient) * float(np.square(L - R).sum())
        preference_label = "disutility_DKK"

    hourly = pd.DataFrame({
        "hour": list(T),
        "reference_load_kWh": R,
        "optimal_load_kWh": L,
        "pv_used_kWh": PV,
        "import_kWh": I,
        "export_kWh": E,
        "absolute_deviation_kWh": absdev,
        "load_minus_reference_kWh": L - R,
    })
    summary = {
        "model": model_kind,
        "coefficient": float(coefficient),
        "objective_total_DKK": objective_val,
        "procurement_cost_DKK": procurement_val,
        preference_label: preference_term,
        "daily_energy_kWh": float(L.sum()),
        "total_absolute_deviation_kWh": float(absdev.sum()) if np.isfinite(absdev).all() else np.nan,
        "hours_deviating_from_reference": int((absdev > TOL).sum()) if np.isfinite(absdev).all() else np.nan,
        "hours_at_load_min_or_max": int(
            ((np.abs(L - data.load_min_kWh) <= TOL) |
             (np.abs(L - data.load_max_kWh) <= TOL)).sum()
        ),
        "daily_import_kWh": float(I.sum()),
        "daily_export_kWh": float(E.sum()),
        "daily_pv_used_kWh": float(PV.sum()),
    }
    return summary, hourly


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # These two case names are used by the Q2 analysis script. Their market and
    # physical data are retained; Q1's objective does not use the reference profile.
    linear_data = load_question("Q2_linear")
    quadratic_data = load_question("Q2_quadratic")

    if linear_data.reference_load is None or quadratic_data.reference_load is None:
        raise ValueError("The Q2 cases must include reference_load data.")
    cL = float(linear_data.linear_disutility)
    cQ = float(quadratic_data.quadratic_disutility)

    configs = [
        ("Q1_linear_utility", linear_data, U_L_Q1),
        ("Q2b_linear_disutility", linear_data, cL),
        ("Q2c_quadratic_disutility", quadratic_data, cQ),
    ]

    summaries, schedules = [], []
    for kind, data, coeff in configs:
        summary, hourly = solve_model(kind, data, coeff)
        summaries.append(summary)
        hourly.insert(0, "model", kind)
        schedules.append(hourly)
        print(
            f"{kind}: objective={summary['objective_total_DKK']:.3f} DKK, "
            f"procurement={summary['procurement_cost_DKK']:.3f} DKK, "
            f"energy={summary['daily_energy_kWh']:.3f} kWh"
        )

    pd.DataFrame(summaries).to_csv(OUT / "q2d_summary.csv", index=False)
    pd.concat(schedules, ignore_index=True).to_csv(OUT / "q2d_hourly_schedules.csv", index=False)
    print(f"\nSaved comparison data to: {OUT}")
    print("Q1 uses U_L_Q1 =", U_L_Q1, "DKK/kWh. Confirm this matches your Q1 base case.")


if __name__ == "__main__":
    main()
