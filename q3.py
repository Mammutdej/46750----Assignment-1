"""Assignment 1, Q3(d) and Q3(e): quadratic disutility + minimum daily energy.

Place in the repository root (next to main.py and q2_analysis.py), then run:
    python q3_analysis.py

The Q3 data folder and the E_min field are found automatically; if that
fails, the error says which name to type into CASE / EMIN_FIELD below.
Outputs go to results/Q3/.
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

# ---- Settings: leave as None to auto-detect, or type the exact names -------
CASE = "Q3"   
EMIN_FIELD = "min_daily_energy_kWh"
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "Q3"
TOL = 1e-5


def find_case():
    """Pick the Q3 data folder (not the battery one) if CASE is not set."""
    if CASE:
        return CASE
    folders = sorted(p.name for p in (ROOT / "data").iterdir() if p.is_dir())
    cands = [f for f in folders if "3" in f and "bat" not in f.lower()]
    for name in cands:
        try:
            load_question(name)
            return name
        except Exception:
            continue
    raise RuntimeError(f"Could not find the Q3 data folder. Folders in data/: {folders}. "
                       "Set CASE at the top of q3_analysis.py.")


def find_emin(d):
    """Find the attribute holding E_min if EMIN_FIELD is not set."""
    if EMIN_FIELD:
        return EMIN_FIELD, float(getattr(d, EMIN_FIELD))
    for name, value in vars(d).items():
        n = name.lower()
        if ("min" in n and ("energy" in n or "daily" in n or n.startswith("e"))
                and "load" not in n and isinstance(value, (int, float)) and value is not None):
            return name, float(value)
    print("Fields in the Q3 data:", vars(d))
    raise RuntimeError("Could not find E_min. Set EMIN_FIELD at the top of q3_analysis.py.")


def solve_q3(d, c_Q: float, E_min):
    """Solve the Q2(c) quadratic model, optionally with sum_t L_t >= E_min.

    E_min=None gives the unconstrained Q2(c) consumer on the same data,
    which is the comparison Q3(e) asks for.
    Returns a summary dict and an hourly DataFrame, including duals.
    """
    m = gp.Model("Q3")
    m.Params.OutputFlag = 0
    T = range(d.n_hours)
    price = np.asarray(d.energy_price, dtype=float)
    p_imp = price + d.import_tariff
    p_exp = price - d.export_tariff
    ref = np.asarray(d.reference_load, dtype=float)
    pv_av = np.asarray(d.pv_available, dtype=float)

    # Same variables and constraints as Q2(c)
    load = m.addVars(T, lb=d.load_min_kWh, ub=d.load_max_kWh, name="load")
    pv = m.addVars(T, lb=0.0, name="pv")
    imp = m.addVars(T, lb=0.0, name="import")
    exp = m.addVars(T, lb=0.0, name="export")
    m.addConstrs((pv[t] <= pv_av[t] for t in T), name="pv_available")
    balance = m.addConstrs(
        (load[t] + exp[t] - imp[t] - pv[t] == 0 for t in T), name="energy_balance"
    )

    # NEW in Q3: the single constraint that couples all 24 hours
    min_energy = None
    if E_min is not None:
        min_energy = m.addConstr(load.sum() >= E_min, name="min_daily_energy")

    procurement = gp.quicksum(
        p_imp[t] * imp[t] - p_exp[t] * exp[t] + d.pv_marginal_cost * pv[t] for t in T
    )
    disutility = gp.quicksum(c_Q * (load[t] - ref[t]) ** 2 for t in T)
    m.setObjective(procurement + disutility, GRB.MINIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi status {m.Status}")

    L = np.array([load[t].X for t in T])
    # Gurobi's Pi = d(objective)/d(right-hand side), minimisation problem:
    #   balance Pi  -> lambda_t (marginal value of 1 kWh in hour t), same sign as report
    #   min_energy Pi (>= constraint) -> mu >= 0, extra cost of raising E_min by 1 kWh
    lam = np.array([-balance[t].Pi for t in T])
    mu = min_energy.Pi if min_energy is not None else 0.0

    hourly = pd.DataFrame({
        "hour": list(T),
        "price": price,
        "p_imp": p_imp,
        "p_exp": p_exp,
        "pv_available": pv_av,
        "reference": ref,
        "load": L,
        "load_minus_ref": L - ref,
        "pv_used": [pv[t].X for t in T],
        "import": [imp[t].X for t in T],
        "export": [exp[t].X for t in T],
        "lambda": lam,
    })
    summary = {
        "E_min": E_min,
        "c_Q": c_Q,
        "objective_DKK": m.ObjVal,
        "procurement_DKK": procurement.getValue(),
        "disutility_DKK": disutility.getValue(),
        "daily_energy_kWh": L.sum(),
        "total_abs_dev_kWh": np.abs(L - ref).sum(),
        "hours_above_ref": int((L - ref > TOL).sum()),
        "mu": mu,
    }
    return summary, hourly


def check_stationarity(d, hourly, c_Q, mu):
    """KKT check (77): in hours where the load is strictly inside its bounds,
    lambda_t + 2 c_Q (L_t - ref_t) must equal mu. Same mu in every such hour."""
    inside = (hourly["load"] > d.load_min_kWh + TOL) & (hourly["load"] < d.load_max_kWh - TOL)
    lhs = hourly["lambda"] + 2 * c_Q * hourly["load_minus_ref"]
    return pd.DataFrame({
        "hour": hourly["hour"][inside],
        "lambda": hourly["lambda"][inside],
        "2cQ(L-ref)": (2 * c_Q * hourly["load_minus_ref"])[inside],
        "sum (should = mu)": lhs[inside],
    })


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    case = find_case()
    d = load_question(case)
    emin_name, E_min = find_emin(d)
    c_Q = float(d.quadratic_disutility)
    print(f"Using data folder '{case}', E_min field '{emin_name}' = {E_min}")

    s_unc, h_unc = solve_q3(d, c_Q, None)   # Q2(c) consumer, same data
    s_q3, h_q3 = solve_q3(d, c_Q, E_min)    # Q3 consumer

    pd.DataFrame([s_unc, s_q3], index=["Q2(c) unconstrained", "Q3 with E_min"]).to_csv(OUT / "base_summary.csv")
    h_q3.assign(load_unconstrained=h_unc["load"]).to_csv(OUT / "base_hourly.csv", index=False)

    # ---- Hypothesis checks for Q3(e) (team's section 3.3.3) ----
    print(f"c_Q = {c_Q}, E_min = {E_min}, sum(ref) = {h_q3['reference'].sum():.2f}")
    print(f"Unconstrained daily energy: {s_unc['daily_energy_kWh']:.2f} kWh")
    print(f"Q3 daily energy:            {s_q3['daily_energy_kWh']:.2f} kWh, mu = {s_q3['mu']:.4f} DKK/kWh")
    print("H1 binding?", abs(s_q3["daily_energy_kWh"] - E_min) < 1e-4 and s_q3["mu"] > TOL)
    print("H2 hours above reference:", list(h_q3.loc[h_q3["load_minus_ref"] > TOL, "hour"]))
    extra = h_q3["load"] - h_unc["load"]
    print("H3/H4 where the extra energy went (kWh added vs unconstrained):")
    print(pd.DataFrame({"hour": h_q3["hour"], "extra_kWh": extra.round(3),
                        "lambda": h_q3["lambda"].round(3)})[extra > TOL].to_string(index=False))
    print("\nKKT stationarity check (77) in interior hours:")
    print(check_stationarity(d, h_q3, c_Q, s_q3["mu"]).round(4).to_string(index=False))

    # ---- Plot: reference vs unconstrained vs Q3 ----
    fig, ax = plt.subplots(figsize=(10, 4.5))
    hrs = h_q3["hour"]
    ax.step(hrs, h_q3["reference"], "k--", where="mid", label="Reference")
    ax.step(hrs, h_unc["load"], where="mid", label="Q2(c), no daily minimum")
    ax.step(hrs, h_q3["load"], where="mid", label=f"Q3, E_min = {E_min:g} kWh")
    ax.bar(hrs, h_q3["pv_available"], alpha=0.2, color="orange", label="PV available")
    ax2 = ax.twinx()
    ax2.step(hrs, h_q3["p_imp"], "r:", where="mid", label="Import price")
    ax2.set_ylabel("DKK/kWh", color="r")
    ax.set(xlabel="Hour", ylabel="kWh", title="Q3(e) base case")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "Q3e_base_schedule.png", dpi=160)
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
