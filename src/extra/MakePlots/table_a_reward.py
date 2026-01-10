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


def _comp_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size == 0:
        return {
            "median_delta": float("nan"),
            "mean_delta": float("nan"),
            "win_pct": float("nan"),
            "tie_pct": float("nan"),
            "mean_delta_win_only": float("nan"),
            "mean_delta_loss_only": float("nan"),
        }

    delta = x - y
    win = x > y
    tie = x == y
    loss = x < y

    delta_win = delta[win]
    delta_loss = delta[loss]

    return {
        "median_delta": float(np.median(delta)),
        "mean_delta": float(delta.mean()),
        "win_pct": 100.0 * float(win.mean()),
        "tie_pct": 100.0 * float(tie.mean()),
        "mean_delta_win_only": float(delta_win.mean()) if delta_win.size else float("nan"),
        "mean_delta_loss_only": float(delta_loss.mean()) if delta_loss.size else float("nan"),
    }


def compute_table_a_reward(dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
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

        ag_gr = _comp_stats(ar, gr)
        ag_de = _comp_stats(ar, dr)
        de_gr = _comp_stats(dr, gr)

        def pack(prefix: str, stats: Dict[str, float]) -> Dict[str, float]:
            return {
                f"{prefix}_median_delta_reward": stats["median_delta"],
                f"{prefix}_mean_delta_reward": stats["mean_delta"],
                f"{prefix}_win_rate_percent": stats["win_pct"],
                f"{prefix}_tie_rate_percent": stats["tie_pct"],
                f"{prefix}_mean_delta_reward_on_wins": stats["mean_delta_win_only"],
                f"{prefix}_mean_delta_reward_on_losses": stats["mean_delta_loss_only"],
            }

        per_seed[seed] = {"n_test": float(len(df))}
        per_seed[seed].update(pack("agent_vs_grevlex", ag_gr))
        per_seed[seed].update(pack("agent_vs_deglex", ag_de))
        per_seed[seed].update(pack("deglex_vs_grevlex", de_gr))

    seeds = sorted(per_seed.keys())
    if not seeds:
        raise ValueError("No seeds with test_metrics available to compute reward Table A.")

    agg: Dict[str, Tuple[float, float]] = {}
    metric_keys = [k for k in per_seed[seeds[0]].keys() if k != "n_test"]
    for k in metric_keys:
        agg[k] = _mean_std([per_seed[s].get(k, float("nan")) for s in seeds])

    return {"seeds": seeds, "n_seeds": len(seeds), "per_seed": per_seed, "agg": agg}


def _fmt_reward(m: float, s: float, fmt_m: str = "{:.6g}", fmt_s: str = "{:.6g}") -> str:
    if not np.isfinite(m):
        return "NA"
    if not np.isfinite(s):
        return f"{fmt_m.format(m)} (reward) +- NA"
    return f"{fmt_m.format(m)} (reward) +- {fmt_s.format(s)} (reward)"


def _fmt_percent(m: float, s: float, fmt_m: str = "{:.3f}", fmt_s: str = "{:.3f}") -> str:
    if not np.isfinite(m):
        return "NA"
    if not np.isfinite(s):
        return f"{fmt_m.format(m)}% +- NA"
    return f"{fmt_m.format(m)}% +- {fmt_s.format(s)}%"


def _print_block(title: str, prefix: str, agg: Dict[str, Tuple[float, float]]) -> None:
    def get(key: str) -> Tuple[float, float]:
        return agg.get(f"{prefix}_{key}", (float("nan"), float("nan")))

    med_m, med_s = get("median_delta_reward")
    mean_m, mean_s = get("mean_delta_reward")
    win_m, win_s = get("win_rate_percent")
    tie_m, tie_s = get("tie_rate_percent")
    w_m, w_s = get("mean_delta_reward_on_wins")
    l_m, l_s = get("mean_delta_reward_on_losses")

    print(f"{title}:")
    print(f"  Median delta reward:                 {_fmt_reward(med_m, med_s)}")
    print(f"  Mean delta reward:                   {_fmt_reward(mean_m, mean_s)}")
    print(f"  Win rate:                            {_fmt_percent(win_m, win_s)}")
    print(f"  Tie rate:                            {_fmt_percent(tie_m, tie_s)}")
    print(f"  Mean delta reward on wins only:      {_fmt_reward(w_m, w_s)}")
    print(f"  Mean delta reward on losses only:    {_fmt_reward(l_m, l_s)}")


def print_table_a_reward(
    table_a: Dict[str, Any],
    include_baseline_sanity: bool = False,
    show_per_seed: bool = False,
) -> None:
    seeds = table_a["seeds"]
    agg = table_a["agg"]

    print("----Table A (REWARD-based, aggregated over seeds)----")
    print(f"Seeds used: {seeds}")

    _print_block("Agent vs GrevLex", "agent_vs_grevlex", agg)
    _print_block("Agent vs DegLex", "agent_vs_deglex", agg)

    if include_baseline_sanity:
        _print_block("DegLex vs GrevLex (sanity check)", "deglex_vs_grevlex", agg)

    print()

    if show_per_seed:
        print("Per-seed breakdown (reward Table A):")
        for s in seeds:
            d = table_a["per_seed"][s]
            print(f"  Seed {s} (n_test={int(d['n_test'])})")

            prefixes = ["agent_vs_grevlex", "agent_vs_deglex"]
            if include_baseline_sanity:
                prefixes.append("deglex_vs_grevlex")

            for pref in prefixes:
                print(
                    f"    {pref}: "
                    f"median_delta={d[pref+'_median_delta_reward']:.6g} (reward), "
                    f"mean_delta={d[pref+'_mean_delta_reward']:.6g} (reward), "
                    f"win={d[pref+'_win_rate_percent']:.3f}%, "
                    f"tie={d[pref+'_tie_rate_percent']:.3f}%, "
                    f"win_mean={d[pref+'_mean_delta_reward_on_wins']:.6g} (reward), "
                    f"loss_mean={d[pref+'_mean_delta_reward_on_losses']:.6g} (reward)"
                )
        print()

