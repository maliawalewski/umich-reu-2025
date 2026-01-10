from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


def _safe_arr(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    x = _safe_arr(np.array(vals, dtype=float))
    if x.size == 0:
        return (float("nan"), float("nan"))
    if x.size == 1:
        return (float(x[0]), 0.0)
    return (float(x.mean()), float(x.std(ddof=1)))


def _comp_stats(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size == 0:
        return {
            "win_rate_percent": float("nan"),
            "tie_rate_percent": float("nan"),
            "mean_improve_percent_on_wins": float("nan"),
            "mean_degrade_percent_on_losses": float("nan"),
            "median_delta_reward": float("nan"),
            "mean_delta_reward": float("nan"),
        }

    delta = x - y
    denom = np.maximum(np.abs(y), eps)
    pct = 100.0 * (delta / denom)

    win = x > y
    tie = x == y
    loss = x < y

    pct_win = pct[win]
    pct_loss = pct[loss]

    return {
        "win_rate_percent": 100.0 * float(win.mean()),
        "tie_rate_percent": 100.0 * float(tie.mean()),
        "mean_improve_percent_on_wins": (
            float(pct_win.mean()) if pct_win.size else float("nan")
        ),
        "mean_degrade_percent_on_losses": (
            float(pct_loss.mean()) if pct_loss.size else float("nan")
        ),
        "median_delta_reward": float(np.median(delta)),
        "mean_delta_reward": float(delta.mean()),
    }


def compute_table_a_reward(
    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]], eps: float = 1e-12
) -> Dict[str, Any]:
    per_seed: Dict[int, Dict[str, float]] = {}

    for seed in sorted(dfs_by_seed.keys()):
        if "test_metrics" not in dfs_by_seed[seed]:
            continue

        df = dfs_by_seed[seed]["test_metrics"].copy()
        required = {"agent_reward", "grevlex_reward", "deglex_reward"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Seed {seed} test_metrics missing columns: {missing}")

        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required))

        ar = df["agent_reward"].to_numpy(float)
        gr = df["grevlex_reward"].to_numpy(float)
        dr = df["deglex_reward"].to_numpy(float)

        ag_gr = _comp_stats(ar, gr, eps=eps)
        ag_de = _comp_stats(ar, dr, eps=eps)
        de_gr = _comp_stats(dr, gr, eps=eps)

        def pack(prefix: str, stats: Dict[str, float]) -> Dict[str, float]:
            out = {
                f"{prefix}_win_rate_percent": stats["win_rate_percent"],
                f"{prefix}_tie_rate_percent": stats["tie_rate_percent"],
                f"{prefix}_mean_improve_percent_on_wins": stats[
                    "mean_improve_percent_on_wins"
                ],
                f"{prefix}_mean_degrade_percent_on_losses": stats[
                    "mean_degrade_percent_on_losses"
                ],
            }
            out.update(
                {
                    f"{prefix}_median_delta_reward": stats["median_delta_reward"],
                    f"{prefix}_mean_delta_reward": stats["mean_delta_reward"],
                }
            )
            return out

        per_seed[seed] = {"n_test": float(len(df))}
        per_seed[seed].update(pack("agent_vs_grevlex", ag_gr))
        per_seed[seed].update(pack("agent_vs_deglex", ag_de))
        per_seed[seed].update(pack("deglex_vs_grevlex", de_gr))

    seeds = sorted(per_seed.keys())
    if not seeds:
        raise ValueError(
            "No seeds with test_metrics available to compute reward Table A."
        )

    agg: Dict[str, Tuple[float, float]] = {}
    metric_keys = [k for k in per_seed[seeds[0]].keys() if k != "n_test"]
    for k in metric_keys:
        agg[k] = _mean_std([per_seed[s].get(k, float("nan")) for s in seeds])

    return {"seeds": seeds, "n_seeds": len(seeds), "per_seed": per_seed, "agg": agg}


def _fmt_percent(
    m: float, s: float, fmt_m: str = "{:.3f}", fmt_s: str = "{:.3f}"
) -> str:
    if not np.isfinite(m):
        return "NA"
    if not np.isfinite(s):
        return f"{fmt_m.format(m)}% +- NA"
    return f"{fmt_m.format(m)}% +- {fmt_s.format(s)}%"


def _fmt_reward(
    m: float, s: float, fmt_m: str = "{:.6g}", fmt_s: str = "{:.6g}"
) -> str:
    if not np.isfinite(m):
        return "NA"
    if not np.isfinite(s):
        return f"{fmt_m.format(m)} (reward) +- NA"
    return f"{fmt_m.format(m)} (reward) +- {fmt_s.format(s)} (reward)"


def _print_block_main(
    title: str,
    prefix: str,
    agg: Dict[str, Tuple[float, float]],
    show_debug_reward_deltas: bool,
) -> None:
    def get(key: str) -> Tuple[float, float]:
        return agg.get(f"{prefix}_{key}", (float("nan"), float("nan")))

    win_m, win_s = get("win_rate_percent")
    tie_m, tie_s = get("tie_rate_percent")
    imp_m, imp_s = get("mean_improve_percent_on_wins")
    deg_m, deg_s = get("mean_degrade_percent_on_losses")

    print(f"{title}:")
    print(f"  Win rate (percent):                   {_fmt_percent(win_m, win_s)}")
    print(f"  Tie rate (percent):                   {_fmt_percent(tie_m, tie_s)}")
    print(f"  Mean improvement on wins only (%):    {_fmt_percent(imp_m, imp_s)}")
    print(f"  Mean degradation on losses only (%):  {_fmt_percent(deg_m, deg_s)}")

    if show_debug_reward_deltas:
        med_m, med_s = get("median_delta_reward")
        mean_m, mean_s = get("mean_delta_reward")
        print(f"  [debug] Median delta reward:          {_fmt_reward(med_m, med_s)}")
        print(f"  [debug] Mean delta reward:            {_fmt_reward(mean_m, mean_s)}")


def print_table_a_reward(
    table_a: Dict[str, Any],
    include_baseline_sanity: bool = False,
    show_per_seed: bool = False,
    show_debug_reward_deltas: bool = False,
) -> None:
    seeds = table_a["seeds"]
    agg = table_a["agg"]

    print("----Table A (MAIN, percent-based; aggregated over seeds)----")
    print(f"Seeds used: {seeds}")

    _print_block_main(
        "Agent vs GrevLex", "agent_vs_grevlex", agg, show_debug_reward_deltas
    )
    _print_block_main(
        "Agent vs DegLex", "agent_vs_deglex", agg, show_debug_reward_deltas
    )

    if include_baseline_sanity:
        _print_block_main(
            "DegLex vs GrevLex (sanity check)",
            "deglex_vs_grevlex",
            agg,
            show_debug_reward_deltas,
        )

    print()

    if show_per_seed:
        print("Per-seed breakdown (percent-based):")
        for s in seeds:
            d = table_a["per_seed"][s]
            print(f"  Seed {s} (n_test={int(d['n_test'])})")

            prefixes = ["agent_vs_grevlex", "agent_vs_deglex"]
            if include_baseline_sanity:
                prefixes.append("deglex_vs_grevlex")

            for pref in prefixes:
                print(
                    f"    {pref}: "
                    f"win={d[pref+'_win_rate_percent']:.3f}%, "
                    f"tie={d[pref+'_tie_rate_percent']:.3f}%, "
                    f"improve_win_only={d[pref+'_mean_improve_percent_on_wins']:.3f}%, "
                    f"degrade_loss_only={d[pref+'_mean_degrade_percent_on_losses']:.3f}%"
                    + (
                        f", [debug] med_delta={d[pref+'_median_delta_reward']:.6g} (reward)"
                        f", mean_delta={d[pref+'_mean_delta_reward']:.6g} (reward)"
                        if show_debug_reward_deltas
                        else ""
                    )
                )
        print()
