from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


ACTION_SCALE_DEFAULT = 1e3


def _to_action_int(weights: np.ndarray, action_scale: float = ACTION_SCALE_DEFAULT) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    a = np.round(action_scale * w).astype(int)
    a = np.maximum(a, 1)
    return a


def compute_weights_table(
    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]],
    action_scale: float = ACTION_SCALE_DEFAULT,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    for seed in sorted(dfs_by_seed.keys()):
        if "final_agent_weight_vector" not in dfs_by_seed[seed]:
            continue

        df = dfs_by_seed[seed]["final_agent_weight_vector"].copy()

        required = {"var", "weight"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Seed {seed} final_agent_weight_vector missing columns: {missing}")

        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["var", "weight"])

        vars_ = [str(v) for v in df["var"].tolist()]
        weights_float = np.asarray(df["weight"].to_numpy(float), dtype=float)

        weights_int = _to_action_int(weights_float, action_scale=action_scale)

        rows.append(
            {
                "seed": seed,
                "vars": vars_,
                "weights_float": weights_float.tolist(),
                "weights_int": weights_int.tolist(),
            }
        )

    seeds = [r["seed"] for r in rows]
    if not seeds:
        raise ValueError("No seeds with final_agent_weight_vector available.")

    return {"seeds": seeds, "rows": rows, "action_scale": float(action_scale)}

def _fmt_vec(v: List[float], fmt: str) -> str:
    return "[" + ", ".join(fmt.format(x) for x in v) + "]"


def print_weights_table(weights_tbl: Dict[str, Any], show_int: bool = True) -> None:
    seeds = weights_tbl["seeds"]
    rows = weights_tbl["rows"]
    action_scale = weights_tbl.get("action_scale", ACTION_SCALE_DEFAULT)

    print("----Final agent weight vectors----")
    print(f"Seeds used: {seeds}")
    if show_int:
        print(f"Columns: seed | vars | weights_float | weights_int (scale={int(action_scale)})")
    else:
        print("Columns: seed | vars | weights_float")
    print()

    for r in rows:
        seed = r["seed"]
        vars_ = r["vars"]
        wf = r["weights_float"]
        line = f"seed={seed} | vars={vars_} | float={_fmt_vec(wf, '{:.6g}')}"
        if show_int:
            wi = r["weights_int"]
            line += f" | int={wi}"
        print(line)

    print()


def weights_table_from_dfs(
    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]],
    action_scale: float = ACTION_SCALE_DEFAULT,
    show_int: bool = True,
) -> Dict[str, Any]:
    tbl = compute_weights_table(dfs_by_seed, action_scale=action_scale)
    print_weights_table(tbl, show_int=show_int)
    return tbl

