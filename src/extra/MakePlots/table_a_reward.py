from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
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

def _median_iqr(vals: List[float]) -> Tuple[float, float, float, float, float]:
    x = _safe_arr(np.array(vals, dtype=float))
    if x.size == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"), 0.0)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return (float(med), float(q1), float(q3), float(q3 - q1), float(x.size))


def _fmt_percent_iqr(med: float, q1: float, q3: float, fmt: str = "{:.3f}") -> str:
    if not np.isfinite(med):
        return "NA"
    return f"{fmt.format(med)}% [IQR {fmt.format(q1)}%, {fmt.format(q3)}%]"


def _fmt_reward_iqr(med: float, q1: float, q3: float, fmt: str = "{:.6g}") -> str:
    if not np.isfinite(med):
        return "NA"
    return f"{fmt.format(med)} (reward) [IQR {fmt.format(q1)}, {fmt.format(q3)}]"


def _comp_stats(
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-12,
    tie_atol: float = 1e-12,
) -> Dict[str, float]:

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    n = int(x.size)

    if n == 0:
        return {
            "n": 0,
            "n_win": 0,
            "n_tie": 0,
            "n_loss": 0,
            "win_rate_percent": float("nan"),
            "tie_rate_percent": float("nan"),
            "loss_rate_percent": float("nan"),
            "mean_improve_percent_on_wins": float("nan"),
            "mean_degrade_percent_on_losses": float("nan"),
            "median_delta_reward": float("nan"),
            "mean_delta_reward": float("nan"),
        }

    delta = x - y
    denom = np.maximum(np.abs(y), eps)
    pct = 100.0 * (delta / denom)

    tie = np.isclose(x, y, atol=tie_atol, rtol=0.0)
    win = (x > y) & (~tie)
    loss = (x < y) & (~tie)

    n_win = int(win.sum())
    n_tie = int(tie.sum())
    n_loss = int(loss.sum())

    pct_win = pct[win]
    pct_loss = pct[loss]

    return {
        "n": n,
        "n_win": n_win,
        "n_tie": n_tie,
        "n_loss": n_loss,
        "win_rate_percent": 100.0 * (n_win / n),
        "tie_rate_percent": 100.0 * (n_tie / n),
        "loss_rate_percent": 100.0 * (n_loss / n),
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
    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]],
    eps: float = 1e-12,
    tie_atol: float = 1e-12,
) -> Dict[str, Any]:

    per_seed: Dict[int, Dict[str, float]] = {}

    pooled_acc: Dict[str, Dict[str, List[np.ndarray]]] = {
        "agent_vs_grevlex": {"x": [], "y": []},
        "agent_vs_deglex": {"x": [], "y": []},
        "deglex_vs_grevlex": {"x": [], "y": []},
    }

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

        ag_gr = _comp_stats(ar, gr, eps=eps, tie_atol=tie_atol)
        ag_de = _comp_stats(ar, dr, eps=eps, tie_atol=tie_atol)
        de_gr = _comp_stats(dr, gr, eps=eps, tie_atol=tie_atol)

        def pack(prefix: str, stats: Dict[str, float]) -> Dict[str, float]:
            out = {
                f"{prefix}_n": float(stats["n"]),
                f"{prefix}_n_win": float(stats["n_win"]),
                f"{prefix}_n_tie": float(stats["n_tie"]),
                f"{prefix}_n_loss": float(stats["n_loss"]),
                f"{prefix}_win_rate_percent": stats["win_rate_percent"],
                f"{prefix}_tie_rate_percent": stats["tie_rate_percent"],
                f"{prefix}_loss_rate_percent": stats["loss_rate_percent"],
                f"{prefix}_mean_improve_percent_on_wins": stats[
                    "mean_improve_percent_on_wins"
                ],
                f"{prefix}_mean_degrade_percent_on_losses": stats[
                    "mean_degrade_percent_on_losses"
                ],
                f"{prefix}_median_delta_reward": stats["median_delta_reward"],
                f"{prefix}_mean_delta_reward": stats["mean_delta_reward"],
            }
            return out

        per_seed[seed] = {"n_test": float(len(df))}
        per_seed[seed].update(pack("agent_vs_grevlex", ag_gr))
        per_seed[seed].update(pack("agent_vs_deglex", ag_de))
        per_seed[seed].update(pack("deglex_vs_grevlex", de_gr))

        pooled_acc["agent_vs_grevlex"]["x"].append(ar)
        pooled_acc["agent_vs_grevlex"]["y"].append(gr)
        pooled_acc["agent_vs_deglex"]["x"].append(ar)
        pooled_acc["agent_vs_deglex"]["y"].append(dr)
        pooled_acc["deglex_vs_grevlex"]["x"].append(dr)
        pooled_acc["deglex_vs_grevlex"]["y"].append(gr)

    seeds = sorted(per_seed.keys())
    if not seeds:
        raise ValueError("No seeds with test_metrics available to compute Table A.")

    by_seed_agg: Dict[str, Tuple[float, float]] = {}
    metric_keys = [k for k in per_seed[seeds[0]].keys() if k != "n_test"]
    for k in metric_keys:
        by_seed_agg[k] = _mean_std([per_seed[s].get(k, float("nan")) for s in seeds])

    pooled: Dict[str, float] = {}
    for pref in ["agent_vs_grevlex", "agent_vs_deglex", "deglex_vs_grevlex"]:
        x_all = (
            np.concatenate(pooled_acc[pref]["x"])
            if pooled_acc[pref]["x"]
            else np.array([], dtype=float)
        )
        y_all = (
            np.concatenate(pooled_acc[pref]["y"])
            if pooled_acc[pref]["y"]
            else np.array([], dtype=float)
        )
        st = _comp_stats(x_all, y_all, eps=eps, tie_atol=tie_atol)

        pooled.update(
            {
                f"{pref}_n": float(st["n"]),
                f"{pref}_n_win": float(st["n_win"]),
                f"{pref}_n_tie": float(st["n_tie"]),
                f"{pref}_n_loss": float(st["n_loss"]),
                f"{pref}_win_rate_percent": st["win_rate_percent"],
                f"{pref}_tie_rate_percent": st["tie_rate_percent"],
                f"{pref}_loss_rate_percent": st["loss_rate_percent"],
                f"{pref}_mean_improve_percent_on_wins": st[
                    "mean_improve_percent_on_wins"
                ],
                f"{pref}_mean_degrade_percent_on_losses": st[
                    "mean_degrade_percent_on_losses"
                ],
                f"{pref}_median_delta_reward": st["median_delta_reward"],
                f"{pref}_mean_delta_reward": st["mean_delta_reward"],
            }
        )

    n_test_total = int(sum(per_seed[s]["n_test"] for s in seeds))

    return {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "n_test_total": n_test_total,
        "per_seed": per_seed,
        "by_seed_agg": by_seed_agg,
        "pooled": pooled,
    }


def _fmt_percent(x: float, fmt: str = "{:.3f}") -> str:
    if not np.isfinite(x):
        return "NA"
    return f"{fmt.format(x)}%"


def _fmt_percent_pm(
    m: float, s: float, fmt_m: str = "{:.3f}", fmt_s: str = "{:.3f}"
) -> str:
    if not np.isfinite(m):
        return "NA"
    if not np.isfinite(s):
        return f"{fmt_m.format(m)}% +- NA"
    return f"{fmt_m.format(m)}% +- {fmt_s.format(s)}%"


def _fmt_reward_pm(
    m: float, s: float, fmt_m: str = "{:.6g}", fmt_s: str = "{:.6g}"
) -> str:
    if not np.isfinite(m):
        return "NA"
    if not np.isfinite(s):
        return f"{fmt_m.format(m)} (reward) +- NA"
    return f"{fmt_m.format(m)} (reward) +- {fmt_s.format(s)} (reward)"


def _print_block_pooled(
    title: str, prefix: str, pooled: Dict[str, float], show_debug_reward_deltas: bool
) -> None:
    n = int(pooled.get(f"{prefix}_n", 0))
    nwin = int(pooled.get(f"{prefix}_n_win", 0))
    ntie = int(pooled.get(f"{prefix}_n_tie", 0))
    nloss = int(pooled.get(f"{prefix}_n_loss", 0))

    win = pooled.get(f"{prefix}_win_rate_percent", float("nan"))
    tie = pooled.get(f"{prefix}_tie_rate_percent", float("nan"))
    loss = pooled.get(f"{prefix}_loss_rate_percent", float("nan"))
    imp = pooled.get(f"{prefix}_mean_improve_percent_on_wins", float("nan"))
    deg = pooled.get(f"{prefix}_mean_degrade_percent_on_losses", float("nan"))

    print(f"{title}:")
    print(f"  n_total={n} (wins={nwin}, ties={ntie}, losses={nloss})")
    print(f"  Win rate (percent):                   {_fmt_percent(win)}")
    print(f"  Tie rate (percent):                   {_fmt_percent(tie)}")
    print(f"  Loss rate (percent):                  {_fmt_percent(loss)}")
    print(f"  Mean improvement on wins only (%):    {_fmt_percent(imp)}")
    print(f"  Mean degradation on losses only (%):  {_fmt_percent(deg)}")

    if show_debug_reward_deltas:
        med = pooled.get(f"{prefix}_median_delta_reward", float("nan"))
        mean = pooled.get(f"{prefix}_mean_delta_reward", float("nan"))
        print(f"  [debug] Median delta reward:          {med:.6g} (reward)")
        print(f"  [debug] Mean delta reward:            {mean:.6g} (reward)")


def _print_block_by_seed_iqr(
    title: str,
    prefix: str,
    per_seed: Dict[int, Dict[str, float]],
    seeds: List[int],
    show_debug_reward_deltas: bool,
) -> None:
    def vals(key: str) -> List[float]:
        return [per_seed[s].get(f"{prefix}_{key}", float("nan")) for s in seeds]

    win_med, win_q1, win_q3, _, _ = _median_iqr(vals("win_rate_percent"))
    tie_med, tie_q1, tie_q3, _, _ = _median_iqr(vals("tie_rate_percent"))
    loss_med, loss_q1, loss_q3, _, _ = _median_iqr(vals("loss_rate_percent"))
    imp_med, imp_q1, imp_q3, _, _ = _median_iqr(vals("mean_improve_percent_on_wins"))
    deg_med, deg_q1, deg_q3, _, _ = _median_iqr(vals("mean_degrade_percent_on_losses"))

    print(f"{title}:")
    print(f"  Win rate (percent):                   {_fmt_percent_iqr(win_med, win_q1, win_q3)}")
    print(f"  Tie rate (percent):                   {_fmt_percent_iqr(tie_med, tie_q1, tie_q3)}")
    print(f"  Loss rate (percent):                  {_fmt_percent_iqr(loss_med, loss_q1, loss_q3)}")
    print(f"  Mean improvement on wins only (%):    {_fmt_percent_iqr(imp_med, imp_q1, imp_q3)}")
    print(f"  Mean degradation on losses only (%):  {_fmt_percent_iqr(deg_med, deg_q1, deg_q3)}")

    if show_debug_reward_deltas:
        med_med, med_q1, med_q3, _, _ = _median_iqr(vals("median_delta_reward"))
        mean_med, mean_q1, mean_q3, _, _ = _median_iqr(vals("mean_delta_reward"))
        print(f"  [debug] Median delta reward:          {_fmt_reward_iqr(med_med, med_q1, med_q3)}")
        print(f"  [debug] Mean delta reward:            {_fmt_reward_iqr(mean_med, mean_q1, mean_q3)}")


def _print_block_by_seed(
    title: str,
    prefix: str,
    agg: Dict[str, Tuple[float, float]],
    show_debug_reward_deltas: bool,
) -> None:
    def get_pm(key: str) -> Tuple[float, float]:
        return agg.get(f"{prefix}_{key}", (float("nan"), float("nan")))

    win_m, win_s = get_pm("win_rate_percent")
    tie_m, tie_s = get_pm("tie_rate_percent")
    loss_m, loss_s = get_pm("loss_rate_percent")
    imp_m, imp_s = get_pm("mean_improve_percent_on_wins")
    deg_m, deg_s = get_pm("mean_degrade_percent_on_losses")

    print(f"{title}:")
    print(f"  Win rate (percent):                   {_fmt_percent_pm(win_m, win_s)}")
    print(f"  Tie rate (percent):                   {_fmt_percent_pm(tie_m, tie_s)}")
    print(f"  Loss rate (percent):                  {_fmt_percent_pm(loss_m, loss_s)}")
    print(f"  Mean improvement on wins only (%):    {_fmt_percent_pm(imp_m, imp_s)}")
    print(f"  Mean degradation on losses only (%):  {_fmt_percent_pm(deg_m, deg_s)}")

    if show_debug_reward_deltas:
        med_m, med_s = get_pm("median_delta_reward")
        mean_m, mean_s = get_pm("mean_delta_reward")
        print(f"  [debug] Median delta reward:          {_fmt_reward_pm(med_m, med_s)}")
        print(
            f"  [debug] Mean delta reward:            {_fmt_reward_pm(mean_m, mean_s)}"
        )


def print_table_a_both_modes(
    table_a: Dict[str, Any],
    include_baseline_sanity: bool = False,
    show_per_seed: bool = True,
    show_debug_reward_deltas: bool = False,
) -> None:
    seeds = table_a["seeds"]

    print("----Table A (MAIN, percent-based; POOLED over all test instances)----")
    print(
        f"Seeds used: {seeds} | n_seeds={table_a['n_seeds']} | n_test_total={table_a['n_test_total']}"
    )
    pooled = table_a["pooled"]

    _print_block_pooled(
        "Agent vs GrevLex", "agent_vs_grevlex", pooled, show_debug_reward_deltas
    )
    _print_block_pooled(
        "Agent vs DegLex", "agent_vs_deglex", pooled, show_debug_reward_deltas
    )
    if include_baseline_sanity:
        _print_block_pooled(
            "DegLex vs GrevLex (sanity check)",
            "deglex_vs_grevlex",
            pooled,
            show_debug_reward_deltas,
        )
    print()

    print("----Table A (APPENDIX; median [IQR] across seeds)----")
    print(f"Seeds used: {seeds} | n_seeds={table_a['n_seeds']}")

    per_seed = table_a["per_seed"]

    _print_block_by_seed_iqr(
        "Agent vs GrevLex", "agent_vs_grevlex", per_seed, seeds, show_debug_reward_deltas
    )
    _print_block_by_seed_iqr(
        "Agent vs DegLex", "agent_vs_deglex", per_seed, seeds, show_debug_reward_deltas
    )
    if include_baseline_sanity:
        _print_block_by_seed_iqr(
            "DegLex vs GrevLex (sanity check)",
            "deglex_vs_grevlex",
            per_seed,
            seeds,
            show_debug_reward_deltas,
        )
    print()

    if include_baseline_sanity:
        _print_block_by_seed(
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
                win = d.get(pref + "_win_rate_percent", float("nan"))
                tie = d.get(pref + "_tie_rate_percent", float("nan"))
                loss = d.get(pref + "_loss_rate_percent", float("nan"))
                imp = d.get(pref + "_mean_improve_percent_on_wins", float("nan"))
                deg = d.get(pref + "_mean_degrade_percent_on_losses", float("nan"))

                msg = (
                    f"    {pref}: "
                    f"win={win:.3f}%, tie={tie:.3f}%, loss={loss:.3f}%, "
                    f"improve_win_only={imp:.3f}%, degrade_loss_only={deg:.3f}%"
                )

                if show_debug_reward_deltas:
                    med = d.get(pref + "_median_delta_reward", float("nan"))
                    mean = d.get(pref + "_mean_delta_reward", float("nan"))
                    msg += f", [debug] med_delta={med:.6g} (reward), mean_delta={mean:.6g} (reward)"

                print(msg)
        print()
