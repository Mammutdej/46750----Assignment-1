"""Optimisation model of a single flexible consumer, implemented with gurobipy.

The class below separates the three steps you will repeat for every question:

    model = FlexibleConsumerModel(data)   # 1. hand over the input data
    model.build()                         # 2. declare variables, objective, constraints
    results = model.solve()               # 3. optimise and collect primal AND dual values

``build()`` is the only method you need to complete for Question 1; the other questions
are variations of it (a different objective, an extra constraint). Copy this file or
subclass ``FlexibleConsumerModel`` and override ``build()`` to keep one model per question.

Two conventions make the dual variables easy to read out afterwards:

* Every constraint family is stored in ``self.con`` under a descriptive name, e.g.
  ``self.con["balance"] = self.m.addConstrs(...)``. ``solve()`` then returns the dual value
  (shadow price, Gurobi attribute ``Pi``) of every constraint in ``self.con`` automatically.
* Bounds that you want a dual for must be written as explicit constraints (``addConstr``),
  not as variable bounds (``lb=``/``ub=``). Gurobi reports the sensitivity of a variable
  bound in the reduced cost (``RC``), not in ``Pi``.
* Duals of quadratic constraints (``m.addQConstr``) are read from ``QCPi`` and require ``QCPDual = 1`` (set below).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from .data_loader import InputData


@dataclass
class Results:
    """Primal and dual solution of one model run."""

    question: str
    status: str
    objective: float
    hourly: pd.DataFrame                   # one row per hour: variables, prices, hourly duals
    duals: dict[str, float] = field(default_factory=dict)   # duals of non-hourly constraints
    meta: dict = field(default_factory=dict)                 # anything else worth keeping (scenario name, ...)

    def save(self, folder: Path | str, tag: str = "") -> None:
        """Write ``hourly`` to CSV and the scalar values to a small text file."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{self.question}{'_' + tag if tag else ''}"
        self.hourly.to_csv(folder / f"{stem}_hourly.csv", index_label="hour")
        with open(folder / f"{stem}_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"status    : {self.status}\nobjective : {self.objective:.4f} DKK\n")
            for k, v in self.duals.items():
                f.write(f"dual[{k}] : {v:.4f}\n")

    def __str__(self) -> str:
        cols = [c for c in self.hourly.columns if not c.startswith("dual_")]
        return (
            f"status: {self.status} | objective: {self.objective:.2f} DKK\n"
            f"daily totals (kWh): " + ", ".join(f"{c}={self.hourly[c].sum():.1f}" for c in cols if c in ("import", "export", "load", "pv"))
            + (f"\nduals: {self.duals}" if self.duals else "")
        )


def _dual(c) -> float:
    """Dual value of a linear (``Pi``) or quadratic (``QCPi``, requires QCPDual=1) constraint."""
    return c.QCPi if isinstance(c, gp.QConstr) else c.Pi


class FlexibleConsumerModel:
    """Consumption problem of one consumer over a 24-hour horizon (Question 1); extend it for Questions 2 and 3."""

    def __init__(self, data: InputData, name: str = "flexible_consumer", verbose: bool = False):
        self.data = data
        self.T = range(data.n_hours)
        self.m = gp.Model(name)
        self.m.Params.OutputFlag = 1 if verbose else 0
        self.m.Params.QCPDual = 1          # only relevant if you add a quadratic constraint (none is needed in Assignment 1)
        self.var: dict[str, gp.tupledict | gp.Var] = {}   # decision variables by name
        self.con: dict[str, gp.tupledict | gp.Constr] = {}  # constraints by name (duals read from here)

    # ------------------------------------------------------------------ 2. build
    def build(self) -> "FlexibleConsumerModel":
        """Build the optimization model for Question 1."""
        d, m, T = self.data, self.m, self.T

        # ------------------------------------------------------------------
        # 1) Decision variables
        # ------------------------------------------------------------------
        # Leave bounds free at the variable level so the explicit constraints
        # below can carry dual information, as the starter project expects.
        self.var["import"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="import")
        self.var["export"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="export")
        self.var["load"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="load")
        self.var["pv"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="pv")

        p_import = self.var["import"]
        p_export = self.var["export"]
        load = self.var["load"]
        pv = self.var["pv"]

        # ------------------------------------------------------------------
        # 2) Objective
        # ------------------------------------------------------------------
        import_cost = gp.quicksum((d.energy_price[t] + d.import_tariff) * p_import[t] for t in T)
        export_revenue = gp.quicksum((d.energy_price[t] - d.export_tariff) * p_export[t] for t in T)
        pv_cost = gp.quicksum(d.pv_marginal_cost * pv[t] for t in T)
        utility = gp.quicksum(d.consumption_utility * load[t] for t in T)

        m.setObjective(import_cost - export_revenue + pv_cost - utility, GRB.MINIMIZE)

        # ------------------------------------------------------------------
        # 3) Constraints
        # ------------------------------------------------------------------
        self.con["balance"] = m.addConstrs(
            (p_import[t] + pv[t] == load[t] + p_export[t] for t in T),
            name="balance",
        )

        self.con["import"] = m.addConstrs(
            (p_import[t] >= 0 for t in T),
            name="import",
        )

        self.con["export"] = m.addConstrs(
            (p_export[t] >= 0 for t in T),
            name="export",
        )

        self.con["pv"] = m.addConstrs(
            (pv[t] >= 0 for t in T),
            name="pv",
        )

        self.con["pv_limit"] = m.addConstrs(
            (pv[t] <= d.pv_available[t] for t in T),
            name="pv_limit",
        )

        self.con["load_min"] = m.addConstrs(
            (load[t] >= d.load_min_kWh for t in T),
            name="load_min",
        )

        self.con["load_max"] = m.addConstrs(
            (load[t] <= d.load_max_kWh for t in T),
            name="load_max",
        )

        if d.max_import_kW is not None:
            self.con["import_cap"] = m.addConstrs(
                (p_import[t] <= d.max_import_kW for t in T),
                name="import_cap",
            )

        if d.max_export_kW is not None:
            self.con["export_cap"] = m.addConstrs(
                (p_export[t] <= d.max_export_kW for t in T),
                name="export_cap",
            )

        m.update()
        return self

    # ------------------------------------------------------------------ 3. solve
    def solve(self) -> Results:
        """Optimise and return primal values, objective and dual values."""
        m = self.m
        m.update()
        if m.NumConstrs == 0 and m.NumQConstrs == 0:
            raise NotImplementedError(
                "The model has no constraints: complete FlexibleConsumerModel.build() in src/model.py first."
            )
        m.optimize()
        status = _status_name(m.Status)
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Optimisation ended with status {status}. Check the model (m.computeIIS() helps for infeasibility).")
        return self._extract_results(status)

    # --------------------------------------------------------------- extraction
    def _extract_results(self, status: str) -> Results:
        d, T = self.data, list(self.T)
        hourly = pd.DataFrame(index=pd.Index(T, name="hour"))
        hourly["price"] = d.energy_price
        hourly["pv_available"] = d.pv_available
        if d.reference_load is not None:
            hourly["reference_load"] = d.reference_load

        # Primal values: every hourly variable family in self.var becomes a column
        for name, v in self.var.items():
            if isinstance(v, gp.tupledict):
                hourly[name] = [v[t].X for t in T]
        scalars = {name: v.X for name, v in self.var.items() if isinstance(v, gp.Var)}

        # Dual values: every constraint family in self.con becomes a 'dual_<name>' column or scalar
        duals: dict[str, float] = {}
        for name, c in self.con.items():
            try:
                if isinstance(c, gp.tupledict):
                    hourly[f"dual_{name}"] = [_dual(c[t]) for t in T]
                else:
                    duals[name] = _dual(c)
            except (AttributeError, gp.GurobiError):
                # No duals available (e.g. model with integer variables)
                pass

        return Results(
            question=d.question,
            status=status,
            objective=self.m.ObjVal,
            hourly=hourly,
            duals=duals,
            meta={"scalar_variables": scalars},
        )



class AbsDeviationModel(FlexibleConsumerModel):
    """Question 2(b): linear disutility of the absolute deviation from the reference load."""

    def build(self) -> "AbsDeviationModel":
        d, m, T = self.data, self.m, self.T
        if d.reference_load is None or d.linear_disutility is None:
            raise ValueError("Q2(b) needs data.reference_load and data.linear_disutility.")

        # ---------------- variables
        self.var["import"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="import")
        self.var["export"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="export")
        self.var["load"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="load")
        self.var["pv"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="pv")
        self.var["delta_pos"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="delta_pos")
        self.var["delta_neg"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="delta_neg")

        p_import, p_export = self.var["import"], self.var["export"]
        load, pv = self.var["load"], self.var["pv"]
        delta_pos, delta_neg = self.var["delta_pos"], self.var["delta_neg"]

        # ---------------- objective (no utility term; disutility instead)
        import_cost = gp.quicksum((d.energy_price[t] + d.import_tariff) * p_import[t] for t in T)
        export_revenue = gp.quicksum((d.energy_price[t] - d.export_tariff) * p_export[t] for t in T)
        pv_cost = gp.quicksum(d.pv_marginal_cost * pv[t] for t in T)
        disutility = gp.quicksum(d.linear_disutility * (delta_pos[t] + delta_neg[t]) for t in T)

        m.setObjective(import_cost - export_revenue + pv_cost + disutility, GRB.MINIMIZE)

        # ---------------- constraints (same as Q1)
        self.con["balance"] = m.addConstrs((p_import[t] + pv[t] == load[t] + p_export[t] for t in T), name="balance")
        self.con["import"] = m.addConstrs((p_import[t] >= 0 for t in T), name="import")
        self.con["export"] = m.addConstrs((p_export[t] >= 0 for t in T), name="export")
        self.con["pv"] = m.addConstrs((pv[t] >= 0 for t in T), name="pv")
        self.con["pv_limit"] = m.addConstrs((pv[t] <= d.pv_available[t] for t in T), name="pv_limit")
        self.con["load_min"] = m.addConstrs((load[t] >= d.load_min_kWh for t in T), name="load_min")
        self.con["load_max"] = m.addConstrs((load[t] <= d.load_max_kWh for t in T), name="load_max")
        if d.max_import_kW is not None:
            self.con["import_cap"] = m.addConstrs((p_import[t] <= d.max_import_kW for t in T), name="import_cap")
        if d.max_export_kW is not None:
            self.con["export_cap"] = m.addConstrs((p_export[t] <= d.max_export_kW for t in T), name="export_cap")

        # ---------------- new in Q2(b): deviation split
        self.con["deviation_link"] = m.addConstrs(
            (load[t] - d.reference_load[t] == delta_pos[t] - delta_neg[t] for t in T),
            name="deviation_link",
        )
        self.con["delta_pos_nonneg"] = m.addConstrs((delta_pos[t] >= 0 for t in T), name="delta_pos_nonneg")
        self.con["delta_neg_nonneg"] = m.addConstrs((delta_neg[t] >= 0 for t in T), name="delta_neg_nonneg")

        m.update()
        return self

class QuadraticDeviationModel(FlexibleConsumerModel):
    """Question 2(c): quadratic disutility around the reference load."""

    def __init__(
        self,
        data: InputData,
        name: str = "quadratic_deviation",
        verbose: bool = False,
        quadratic_coefficient: float | None = None,
    ):
        super().__init__(data, name=name, verbose=verbose)
        self.quadratic_coefficient = (
            data.quadratic_disutility
            if quadratic_coefficient is None
            else float(quadratic_coefficient)
        )

    def build(self) -> "QuadraticDeviationModel":
        d, m, T = self.data, self.m, self.T

        if d.reference_load is None or self.quadratic_coefficient is None:
            raise ValueError(
                "Q2(c) needs data.reference_load "
                "and data.quadratic_disutility."
            )

        if self.quadratic_coefficient <= 0:
            raise ValueError(
                "The quadratic disutility coefficient must be positive."
            )

        # --------------------------------------------------------------
        # 1) Decision variables
        # --------------------------------------------------------------
        self.var["import"] = m.addVars(
            T,
            lb=-GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="import",
        )
        self.var["export"] = m.addVars(
            T,
            lb=-GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="export",
        )
        self.var["load"] = m.addVars(
            T,
            lb=-GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="load",
        )
        self.var["pv"] = m.addVars(
            T,
            lb=-GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="pv",
        )

        p_import = self.var["import"]
        p_export = self.var["export"]
        load = self.var["load"]
        pv = self.var["pv"]

        # --------------------------------------------------------------
        # 2) Objective function
        # --------------------------------------------------------------
        import_cost = gp.quicksum(
            (d.energy_price[t] + d.import_tariff) * p_import[t]
            for t in T
        )

        export_revenue = gp.quicksum(
            (d.energy_price[t] - d.export_tariff) * p_export[t]
            for t in T
        )

        pv_cost = gp.quicksum(
            d.pv_marginal_cost * pv[t]
            for t in T
        )

        quadratic_disutility = gp.quicksum(
            self.quadratic_coefficient
            * (load[t] - d.reference_load[t])
            * (load[t] - d.reference_load[t])
            for t in T
        )

        m.setObjective(
            import_cost
            - export_revenue
            + pv_cost
            + quadratic_disutility,
            GRB.MINIMIZE,
        )

        # --------------------------------------------------------------
        # 3) Constraints
        # --------------------------------------------------------------
        self.con["balance"] = m.addConstrs(
            (
                p_import[t] + pv[t]
                == load[t] + p_export[t]
                for t in T
            ),
            name="balance",
        )

        self.con["import"] = m.addConstrs(
            (p_import[t] >= 0 for t in T),
            name="import",
        )

        self.con["export"] = m.addConstrs(
            (p_export[t] >= 0 for t in T),
            name="export",
        )

        self.con["pv"] = m.addConstrs(
            (pv[t] >= 0 for t in T),
            name="pv",
        )

        self.con["pv_limit"] = m.addConstrs(
            (pv[t] <= d.pv_available[t] for t in T),
            name="pv_limit",
        )

        self.con["load_min"] = m.addConstrs(
            (load[t] >= d.load_min_kWh for t in T),
            name="load_min",
        )

        self.con["load_max"] = m.addConstrs(
            (load[t] <= d.load_max_kWh for t in T),
            name="load_max",
        )

        if d.max_import_kW is not None:
            self.con["import_cap"] = m.addConstrs(
                (
                    p_import[t] <= d.max_import_kW
                    for t in T
                ),
                name="import_cap",
            )

        if d.max_export_kW is not None:
            self.con["export_cap"] = m.addConstrs(
                (
                    p_export[t] <= d.max_export_kW
                    for t in T
                ),
                name="export_cap",
            )

        m.update()
        return self


class MinimumEnergyQuadraticModel(QuadraticDeviationModel):
    """Question 3: quadratic disutility with a minimum daily energy requirement."""

    def __init__(
        self,
        data: InputData,
        minimum_daily_energy: float | None,
        name: str = "minimum_energy_quadratic",
        verbose: bool = False,
        quadratic_coefficient: float | None = None,
    ):
        super().__init__(
            data,
            name=name,
            verbose=verbose,
            quadratic_coefficient=quadratic_coefficient,
        )
        self.minimum_daily_energy = (
            None if minimum_daily_energy is None else float(minimum_daily_energy)
        )

    def build(self) -> "MinimumEnergyQuadraticModel":
        super().build()
        if self.minimum_daily_energy is not None:
            self.con["min_daily_energy"] = self.m.addConstr(
                self.var["load"].sum() >= self.minimum_daily_energy,
                name="min_daily_energy",
            )
        self.m.update()
        return self

class BatteryModel(MinimumEnergyQuadraticModel):
    """Question 3(g): minimum-energy quadratic model with a battery."""

    def build(self) -> "BatteryModel":
        d, m, T = self.data, self.m, self.T

        # First build the model from Question 3(a).
        super().build()

        # Check that the selected dataset contains battery data.
        battery_parameters = [
            d.battery_capacity_kWh,
            d.battery_max_charge_kW,
            d.battery_max_discharge_kW,
            d.battery_charging_efficiency,
            d.battery_discharging_efficiency,
            d.battery_initial_soc_kWh,
        ]

        if any(value is None for value in battery_parameters):
            raise ValueError(
                "BatteryModel requires the Q3_battery dataset."
            )

        # --------------------------------------------------------------
        # 1) Remove the old power balance
        # --------------------------------------------------------------
        # The balance from Question 3(a) does not include the battery,
        # so it must be replaced by a new balance constraint.
        old_balance = self.con.pop("balance")
        m.remove(list(old_balance.values()))
        m.update()

        # --------------------------------------------------------------
        # 2) Add battery decision variables
        # --------------------------------------------------------------
        self.var["charge"] = m.addVars(
            T,
            lb=-GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="charge",
        )

        self.var["discharge"] = m.addVars(
            T,
            lb=-GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="discharge",
        )

        # soc[t] is the state of charge at the end of hour t.
        self.var["soc"] = m.addVars(
            T,
            lb=-GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="soc",
        )

        p_import = self.var["import"]
        p_export = self.var["export"]
        load = self.var["load"]
        pv = self.var["pv"]
        charge = self.var["charge"]
        discharge = self.var["discharge"]
        soc = self.var["soc"]

        # --------------------------------------------------------------
        # 3) New power balance including the battery
        # --------------------------------------------------------------
        self.con["balance"] = m.addConstrs(
            (
                p_import[t] + pv[t] + discharge[t]
                == load[t] + p_export[t] + charge[t]
                for t in T
            ),
            name="balance",
        )

        # --------------------------------------------------------------
        # 4) Charging and discharging limits
        # --------------------------------------------------------------
        self.con["charge_lower"] = m.addConstrs(
            (charge[t] >= 0 for t in T),
            name="charge_lower",
        )

        self.con["charge_upper"] = m.addConstrs(
            (
                charge[t] <= d.battery_max_charge_kW
                for t in T
            ),
            name="charge_upper",
        )

        self.con["discharge_lower"] = m.addConstrs(
            (discharge[t] >= 0 for t in T),
            name="discharge_lower",
        )

        self.con["discharge_upper"] = m.addConstrs(
            (
                discharge[t] <= d.battery_max_discharge_kW
                for t in T
            ),
            name="discharge_upper",
        )

        # --------------------------------------------------------------
        # 5) State-of-charge dynamics
        # --------------------------------------------------------------
        self.con["soc_dynamics"] = m.addConstrs(
            (
                soc[t]
                == (
                    d.battery_initial_soc_kWh
                    if t == 0
                    else soc[t - 1]
                )
                + d.battery_charging_efficiency * charge[t]
                - discharge[t] / d.battery_discharging_efficiency
                for t in T
            ),
            name="soc_dynamics",
        )

        # --------------------------------------------------------------
        # 6) State-of-charge limits
        # --------------------------------------------------------------
        self.con["soc_lower"] = m.addConstrs(
            (soc[t] >= 0 for t in T),
            name="soc_lower",
        )

        self.con["soc_upper"] = m.addConstrs(
            (
                soc[t] <= d.battery_capacity_kWh
                for t in T
            ),
            name="soc_upper",
        )

        # --------------------------------------------------------------
        # 7) End-of-horizon condition
        # --------------------------------------------------------------
        # The battery must finish the day with the same energy with
        # which it started.
        self.con["terminal_soc"] = m.addConstr(
            soc[d.n_hours - 1] == d.battery_initial_soc_kWh,
            name="terminal_soc",
        )

        m.update()
        return self
    
_STATUS = {
    GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE", GRB.UNBOUNDED: "UNBOUNDED",
    GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.NUMERIC: "NUMERIC", GRB.INTERRUPTED: "INTERRUPTED",
}


def _status_name(code: int) -> str:
    return _STATUS.get(code, f"STATUS_{code}")
