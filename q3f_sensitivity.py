"""Assignment 1, Q3(f): sensitivity analysis.

Run from the repository root, AFTER the Q3 script (q3_analysis.py or q3.py) works:
    python q3f_sensitivity.py

Three one-at-a-time sweeps around the base case:
  1. E_min        (how big the daily duty is)
  2. c_Q          (how much the consumer dislikes deviating)
  3. price spread (how different cheap and expensive hours are)
Outputs: results/Q3f/*.csv and *.png, plus sanity checks printed to screen.
"""
import copy
import dataclasses
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data_loader import load_question
try:  # the Q3 script may be saved as q3_analysis.py or q3.py
    from q3_analysis import solve_q3, find_case, find_emin, ROOT, TOL
except ModuleNotFoundError:
    from q3 import solve_q3, find_case, find_emin, ROOT, TOL

OUT = ROOT / "results" / "Q3f"


def sweep(d, values, make_case):
    """make_case(value) -> (data, c_Q, E_min). Returns summary table and hourly loads."""
    rows, loads = [], {}
    for v in values:
        dv, cq, em = make_case(v)
        s, h = solve_q3(dv, cq, em)
        s["value"] = v
        rows.append(s)
        loads[v] = h["load"].to_numpy()
    return pd.DataFrame(rows), loads


def plot_sweep(table, loads, ref, xlabel, name, show=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.plot(table["value"], table["mu"], "o-", label="μ (DKK/kWh)")
    ax.set(xlabel=xlabel, ylabel="μ (DKK/kWh)", title="Price of the daily minimum")
    ax2 = ax.twinx()
    ax2.plot(table["value"], table["hours_above_ref"], "s--", color="gray", label="hours above ref")
    ax2.set_ylabel("hours with load above reference")
    ax = axes[1]
    ax.step(range(len(ref)), ref, "k--", where="mid", label="reference")
    for v in (show or list(loads)[:: max(1, len(loads) // 4)]):
        ax.step(range(len(ref)), loads[v], where="mid", label=f"{xlabel} = {v:g}")
    ax.set(xlabel="hour", ylabel="load (kWh)", title="Load profiles")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_question(find_case())
    cq0 = float(d.quadratic_disutility)
    em0 = find_emin(d)[1]
    ref = np.asarray(d.reference_load, dtype=float)
    s_unc, _ = solve_q3(d, cq0, None)
    E_unc = s_unc["daily_energy_kWh"]
    print(f"Base: c_Q={cq0}, E_min={em0}, unconstrained daily energy E_unc={E_unc:.2f}")

    # ---- 1. E_min sweep: from clearly non-binding to well above the reference total
    # 5 kWh grid, plus finer points where behaviour changes (around the
    # reference total 32.34 kWh). E_unc itself is left out: a point a few Wh
    # above the kink is binding only within solver tolerance.
    em_values = sorted(set(np.round(np.r_[np.arange(10, 81, 5), 31, 32, float(ref.sum()), 33, 34, em0], 2)))
    em_values = [v for v in em_values if v <= 24 * d.load_max_kWh]  # stay feasible
    t1, l1 = sweep(d, em_values, lambda v: (d, cq0, v))
    t1.to_csv(OUT / "sweep_Emin.csv", index=False)
    plot_sweep(t1, l1, ref, "E_min", "sweep_Emin")

    # ---- 2. c_Q sweep: factors around the base value
    cq_values = [round(cq0 * f, 4) for f in (0.1, 0.25, 0.5, 1, 2, 4, 10)]
    t2, l2 = sweep(d, cq_values, lambda v: (d, v, em0))
    t2.to_csv(OUT / "sweep_cQ.csv", index=False)
    plot_sweep(t2, l2, ref, "c_Q", "sweep_cQ")

    # ---- 3. price-spread sweep: p_t' = mean + k (p_t - mean); k=1 is the base case
    price = np.asarray(d.energy_price, dtype=float)
    def spread_case(k):
        new_price = price.mean() + k * (price - price.mean())
        try:  # works if InputData is a (possibly frozen) dataclass
            dk = dataclasses.replace(d, energy_price=new_price)
        except TypeError:
            dk = copy.copy(d)
            dk.energy_price = new_price
        return dk, cq0, em0
    k_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    t3, l3 = sweep(d, k_values, spread_case)
    t3.to_csv(OUT / "sweep_spread.csv", index=False)
    plot_sweep(t3, l3, ref, "spread k", "sweep_spread", show=k_values)

    # ---- Sanity checks: each should print True --------------------------
    print("\nSANITY CHECKS (each should be True)")
    # skip points within 0.1 kWh of the kink at E_unc (numerically ambiguous)
    below = t1[t1["value"] < E_unc - 0.1]
    above = t1[t1["value"] > E_unc + 0.1]
    print("1. E_min below E_unc -> rule unused, mu = 0:",
          bool((below["mu"].abs() < 1e-6).all()))
    print("2. E_min above E_unc -> rule binding, energy = E_min:",
          bool((abs(above["daily_energy_kWh"] - above["value"]) < 1e-4).all()))
    print("   (slope of mu between runs; theory: 2 c_Q / number of interior hours)")
    print(pd.DataFrame({"E_min": t1["value"], "mu": t1["mu"].round(4),
                        "dmu/dE": t1["mu"].diff().div(t1["value"].diff()).round(4),
                        "hours_above_ref": t1["hours_above_ref"]}).to_string(index=False))
    print("3. mu never decreases as E_min grows:",
          bool((np.diff(t1["mu"]) >= -1e-6).all()))
    # 4. mu should equal the slope of the optimal cost w.r.t. E_min (definition of a dual)
    e = em0
    up, _ = solve_q3(d, cq0, e + 0.01)
    dn, _ = solve_q3(d, cq0, e - 0.01)
    slope = (up["objective_DKK"] - dn["objective_DKK"]) / 0.02
    s_base, _ = solve_q3(d, cq0, e)
    print(f"4. mu = {s_base['mu']:.4f}, numerical slope of cost = {slope:.4f}:",
          abs(slope - s_base["mu"]) < 1e-2)
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()