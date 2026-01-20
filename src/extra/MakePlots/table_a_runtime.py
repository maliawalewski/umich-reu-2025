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


def _pct(a: np.ndarray, q: float) -> float:
    a = _safe_arr(a)
    if a.size == 0:
        return float("nan")
    return float(np.percentile(a, q))


def _topk_loss_outliers(
    x_time: np.ndarray,
    y_time: np.ndarray,
    loss_mask: np.ndarray,
    *,
    eps: float,
    k: int = 5,
) -> List[Dict[str, float]]:
    """
    Return top-k *largest* ratios (x/y) among loss cases (x > y).
    Each item: ratio, agent_time, base_time, delta_s, pct_slowdown.
    """
    x = np.asarray(x_time, dtype=float)
    y = np.asarray(y_time, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & loss_mask
    x = x[ok]
    y = y[ok]
    if x.size == 0:
        return []

    ratio = x / np.maximum(y, eps)
    # slowdown percent (negative): 100*(y-x)/y = 100*(1 - ratio)
    pct_slow = 100.0 * (1.0 - ratio)

    idx = np.argsort(ratio)[-k:][::-1]
    out: List[Dict[str, float]] = []
    for i in idx:
        out.append(
            {
                "ratio": float(ratio[i]),
                "agent_time_s": float(x[i]),
                "base_time_s": float(y[i]),
                "delta_s": float(x[i] - y[i]),
                "pct_slowdown": float(pct_slow[i]),
            }
        )
    return out


def _comp_stats_time(
    x_time: np.ndarray,
    y_time: np.ndarray,
    eps: float = 1e-12,
    tie_atol: float = 1e-12,
) -> Dict[str, float]:
    """
    Compare x vs y where smaller time is better.
      - "win" means x < y (x faster)
      - percent speedup: 100 * (y - x) / max(|y|, eps)
        (positive => speedup, negative => slowdown)
    Adds loss-only diagnostics to explain giant negative slowdowns caused by tiny y.
    """
    x = np.asarray(x_time, dtype=float)
    y = np.asarray(y_time, dtype=float)

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
            "median_delta_time_s": float("nan"),
            "mean_delta_time_s": float("nan"),
            "loss_ratio_p50": float("nan"),
            "loss_ratio_p90": float("nan"),
            "loss_ratio_p99": float("nan"),
            "loss_ratio_max": float("nan"),
            "loss_base_time_min_s": float("nan"),
            "loss_base_time_p01_s": float("nan"),
            "loss_base_time_p05_s": float("nan"),
            "loss_base_time_p50_s": float("nan"),
            "loss_ratio_ge_10_percent": float("nan"),
            "loss_ratio_ge_50_percent": float("nan"),
        }

    delta = x - y
    denom = np.maximum(np.abs(y), eps)
    pct = 100.0 * ((y - x) / denom)

    tie = np.isclose(x, y, atol=tie_atol, rtol=0.0)
    win = (x < y) & (~tie)
    loss = (x > y) & (~tie)

    n_win = int(win.sum())
    n_tie = int(tie.sum())
    n_loss = int(loss.sum())

    pct_win = pct[win]
    pct_loss = pct[loss]

    if n_loss > 0:
        loss_ratio = x[loss] / np.maximum(y[loss], eps)
        loss_base = y[loss]
    else:
        loss_ratio = np.array([], dtype=float)
        loss_base = np.array([], dtype=float)

    loss_ratio_p50 = _pct(loss_ratio, 50.0)
    loss_ratio_p90 = _pct(loss_ratio, 90.0)
    loss_ratio_p99 = _pct(loss_ratio, 99.0)
    loss_ratio_max = float(np.max(loss_ratio)) if loss_ratio.size else float("nan")

    loss_base_min = float(np.min(loss_base)) if loss_base.size else float("nan")
    loss_base_p01 = _pct(loss_base, 1.0)
    loss_base_p05 = _pct(loss_base, 5.0)
    loss_base_p50 = _pct(loss_base, 50.0)

    if loss_ratio.size:
        loss_ratio_ge_10 = 100.0 * float(np.mean(loss_ratio >= 10.0))
        loss_ratio_ge_50 = 100.0 * float(np.mean(loss_ratio >= 50.0))
    else:
        loss_ratio_ge_10 = float("nan")
        loss_ratio_ge_50 = float("nan")

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
        "median_delta_time_s": float(np.median(delta)),
        "mean_delta_time_s": float(delta.mean()),
        "loss_ratio_p50": float(loss_ratio_p50),
        "loss_ratio_p90": float(loss_ratio_p90),
        "loss_ratio_p99": float(loss_ratio_p99),
        "loss_ratio_max": float(loss_ratio_max),
        "loss_base_time_min_s": float(loss_base_min),
        "loss_base_time_p01_s": float(loss_base_p01),
        "loss_base_time_p05_s": float(loss_base_p05),
        "loss_base_time_p50_s": float(loss_base_p50),
        "loss_ratio_ge_10_percent": float(loss_ratio_ge_10),
        "loss_ratio_ge_50_percent": float(loss_ratio_ge_50),
    }


def compute_table_a_runtime(
    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]],
    eps: float = 1e-12,
    tie_atol: float = 1e-12,
) -> Dict[str, Any]:
    outliers_per_seed: Dict[int, Dict[str, List[Dict[str, float]]]] = {}
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

        required = {"agent_time_s", "grevlex_time_s", "deglex_time_s"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Seed {seed} test_metrics missing columns: {missing}")

        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required))

        at = df["agent_time_s"].to_numpy(float)
        gt = df["grevlex_time_s"].to_numpy(float)
        dt = df["deglex_time_s"].to_numpy(float)

        ag_gr = _comp_stats_time(at, gt, eps=eps, tie_atol=tie_atol)
        ag_de = _comp_stats_time(at, dt, eps=eps, tie_atol=tie_atol)
        de_gr = _comp_stats_time(dt, gt, eps=eps, tie_atol=tie_atol)

        loss_ag_de = (at > dt) & (~np.isclose(at, dt, atol=tie_atol, rtol=0.0))
        loss_ag_gr = (at > gt) & (~np.isclose(at, gt, atol=tie_atol, rtol=0.0))

        outliers_per_seed.setdefault(seed, {})
        outliers_per_seed[seed]["agent_vs_deglex"] = _topk_loss_outliers(
            at, dt, loss_ag_de, eps=eps, k=5
        )
        outliers_per_seed[seed]["agent_vs_grevlex"] = _topk_loss_outliers(
            at, gt, loss_ag_gr, eps=eps, k=5
        )

        def pack(prefix: str, stats: Dict[str, float]) -> Dict[str, float]:
            keys = [
                "n",
                "n_win",
                "n_tie",
                "n_loss",
                "win_rate_percent",
                "tie_rate_percent",
                "loss_rate_percent",
                "mean_improve_percent_on_wins",
                "mean_degrade_percent_on_losses",
                "median_delta_time_s",
                "mean_delta_time_s",
                "loss_ratio_p50",
                "loss_ratio_p90",
                "loss_ratio_p99",
                "loss_ratio_max",
                "loss_base_time_min_s",
                "loss_base_time_p01_s",
                "loss_base_time_p05_s",
                "loss_base_time_p50_s",
                "loss_ratio_ge_10_percent",
                "loss_ratio_ge_50_percent",
            ]
            out: Dict[str, float] = {}
            for k in keys:
                out[f"{prefix}_{k}"] = float(stats.get(k, float("nan")))
            return out

        per_seed[seed] = {"n_test": float(len(df))}
        per_seed[seed].update(pack("agent_vs_grevlex", ag_gr))
        per_seed[seed].update(pack("agent_vs_deglex", ag_de))
        per_seed[seed].update(pack("deglex_vs_grevlex", de_gr))

        pooled_acc["agent_vs_grevlex"]["x"].append(at)
        pooled_acc["agent_vs_grevlex"]["y"].append(gt)
        pooled_acc["agent_vs_deglex"]["x"].append(at)
        pooled_acc["agent_vs_deglex"]["y"].append(dt)
        pooled_acc["deglex_vs_grevlex"]["x"].append(dt)
        pooled_acc["deglex_vs_grevlex"]["y"].append(gt)

    seeds = sorted(per_seed.keys())
    if not seeds:
        raise ValueError(
            "No seeds with test_metrics available to compute Table A (runtime)."
        )

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

        st = _comp_stats_time(x_all, y_all, eps=eps, tie_atol=tie_atol)

        keys = [
            "n",
            "n_win",
            "n_tie",
            "n_loss",
            "win_rate_percent",
            "tie_rate_percent",
            "loss_rate_percent",
            "mean_improve_percent_on_wins",
            "mean_degrade_percent_on_losses",
            "median_delta_time_s",
            "mean_delta_time_s",
            "loss_ratio_p50",
            "loss_ratio_p90",
            "loss_ratio_p99",
            "loss_ratio_max",
            "loss_base_time_min_s",
            "loss_base_time_p01_s",
            "loss_base_time_p05_s",
            "loss_base_time_p50_s",
            "loss_ratio_ge_10_percent",
            "loss_ratio_ge_50_percent",
        ]
        for k in keys:
            pooled[f"{pref}_{k}"] = float(st.get(k, float("nan")))

    pooled_outliers: Dict[str, List[Dict[str, float]]] = {}
    for pref in ["agent_vs_grevlex", "agent_vs_deglex"]:
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
        loss_mask = (x_all > y_all) & (
            ~np.isclose(x_all, y_all, atol=tie_atol, rtol=0.0)
        )
        pooled_outliers[pref] = _topk_loss_outliers(
            x_all, y_all, loss_mask, eps=eps, k=10
        )

    n_test_total = int(sum(per_seed[s]["n_test"] for s in seeds))

    return {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "n_test_total": n_test_total,
        "per_seed": per_seed,
        "by_seed_agg": by_seed_agg,
        "pooled": pooled,
        "outliers_per_seed": outliers_per_seed,
        "pooled_outliers": pooled_outliers,
    }


def _fmt_time_pm(
    m: float, s: float, fmt_m: str = "{:.6g}", fmt_s: str = "{:.6g}"
) -> str:
    if not np.isfinite(m):
        return "NA"
    if not np.isfinite(s):
        return f"{fmt_m.format(m)} (s) +- NA"
    return f"{fmt_m.format(m)} (s) +- {fmt_s.format(s)} (s)"


def _fmt_time_iqr(med: float, q1: float, q3: float, fmt: str = "{:.6g}") -> str:
    if not np.isfinite(med):
        return "NA"
    return f"{fmt.format(med)} (s) [IQR {fmt.format(q1)}, {fmt.format(q3)}]"


def _print_loss_diagnostics(prefix: str, pooled: Dict[str, float]) -> None:
    lr50 = pooled.get(f"{prefix}_loss_ratio_p50", float("nan"))
    lr90 = pooled.get(f"{prefix}_loss_ratio_p90", float("nan"))
    lr99 = pooled.get(f"{prefix}_loss_ratio_p99", float("nan"))
    lrmax = pooled.get(f"{prefix}_loss_ratio_max", float("nan"))

    bmin = pooled.get(f"{prefix}_loss_base_time_min_s", float("nan"))
    bp01 = pooled.get(f"{prefix}_loss_base_time_p01_s", float("nan"))
    bp05 = pooled.get(f"{prefix}_loss_base_time_p05_s", float("nan"))
    bp50 = pooled.get(f"{prefix}_loss_base_time_p50_s", float("nan"))

    ge10 = pooled.get(f"{prefix}_loss_ratio_ge_10_percent", float("nan"))
    ge50 = pooled.get(f"{prefix}_loss_ratio_ge_50_percent", float("nan"))

    print(
        f"  [loss diag] ratio agent/base (p50/p90/p99/max): "
        f"{lr50:.3g}, {lr90:.3g}, {lr99:.3g}, {lrmax:.3g}"
    )
    print(
        f"  [loss diag] baseline time on losses (min/p1/p5/med) s: "
        f"{bmin:.3g}, {bp01:.3g}, {bp05:.3g}, {bp50:.3g}"
    )
    print(
        f"  [loss diag] % of losses with ratio >=10x: {ge10:.3f}%, >=50x: {ge50:.3f}%"
    )


def _print_block_pooled_time(
    title: str, prefix: str, pooled: Dict[str, float], show_debug_time_deltas: bool
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
    print(f"  n_total={n} (faster={nwin}, ties={ntie}, slower={nloss})")
    print(f"  Faster rate (percent):                {_fmt_percent(win)}")
    print(f"  Tie rate (percent):                   {_fmt_percent(tie)}")
    print(f"  Slower rate (percent):                {_fmt_percent(loss)}")
    print(f"  Mean speedup on faster only (%):      {_fmt_percent(imp)}")
    print(f"  Mean slowdown on slower only (%):     {_fmt_percent(deg)}")

    _print_loss_diagnostics(prefix, pooled)

    if show_debug_time_deltas:
        med = pooled.get(f"{prefix}_median_delta_time_s", float("nan"))
        mean = pooled.get(f"{prefix}_mean_delta_time_s", float("nan"))
        print(
            f"  [debug] Median delta time:            {med:.6g} (s)  (agent - baseline)"
        )
        print(
            f"  [debug] Mean delta time:              {mean:.6g} (s)  (agent - baseline)"
        )


def _print_block_by_seed_iqr_time(
    title: str,
    prefix: str,
    per_seed: Dict[int, Dict[str, float]],
    seeds: List[int],
    show_debug_time_deltas: bool,
) -> None:
    def vals(key: str) -> List[float]:
        return [per_seed[s].get(f"{prefix}_{key}", float("nan")) for s in seeds]

    win_med, win_q1, win_q3, _, _ = _median_iqr(vals("win_rate_percent"))
    tie_med, tie_q1, tie_q3, _, _ = _median_iqr(vals("tie_rate_percent"))
    loss_med, loss_q1, loss_q3, _, _ = _median_iqr(vals("loss_rate_percent"))
    imp_med, imp_q1, imp_q3, _, _ = _median_iqr(vals("mean_improve_percent_on_wins"))
    deg_med, deg_q1, deg_q3, _, _ = _median_iqr(vals("mean_degrade_percent_on_losses"))

    print(f"{title}:")
    print(
        f"  Faster rate (percent):                {_fmt_percent_iqr(win_med, win_q1, win_q3)}"
    )
    print(
        f"  Tie rate (percent):                   {_fmt_percent_iqr(tie_med, tie_q1, tie_q3)}"
    )
    print(
        f"  Slower rate (percent):                {_fmt_percent_iqr(loss_med, loss_q1, loss_q3)}"
    )
    print(
        f"  Mean speedup on faster only (%):      {_fmt_percent_iqr(imp_med, imp_q1, imp_q3)}"
    )
    print(
        f"  Mean slowdown on slower only (%):     {_fmt_percent_iqr(deg_med, deg_q1, deg_q3)}"
    )

    if show_debug_time_deltas:
        med_med, med_q1, med_q3, _, _ = _median_iqr(vals("median_delta_time_s"))
        mean_med, mean_q1, mean_q3, _, _ = _median_iqr(vals("mean_delta_time_s"))
        print(
            f"  [debug] Median delta time:            {_fmt_time_iqr(med_med, med_q1, med_q3)}  (agent - baseline)"
        )
        print(
            f"  [debug] Mean delta time:              {_fmt_time_iqr(mean_med, mean_q1, mean_q3)}  (agent - baseline)"
        )


def _print_block_by_seed_time(
    title: str,
    prefix: str,
    agg: Dict[str, Tuple[float, float]],
    show_debug_time_deltas: bool,
) -> None:
    def get_pm(key: str) -> Tuple[float, float]:
        return agg.get(f"{prefix}_{key}", (float("nan"), float("nan")))

    win_m, win_s = get_pm("win_rate_percent")
    tie_m, tie_s = get_pm("tie_rate_percent")
    loss_m, loss_s = get_pm("loss_rate_percent")
    imp_m, imp_s = get_pm("mean_improve_percent_on_wins")
    deg_m, deg_s = get_pm("mean_degrade_percent_on_losses")

    print(f"{title}:")
    print(f"  Faster rate (percent):                {_fmt_percent_pm(win_m, win_s)}")
    print(f"  Tie rate (percent):                   {_fmt_percent_pm(tie_m, tie_s)}")
    print(f"  Slower rate (percent):                {_fmt_percent_pm(loss_m, loss_s)}")
    print(f"  Mean speedup on faster only (%):      {_fmt_percent_pm(imp_m, imp_s)}")
    print(f"  Mean slowdown on slower only (%):     {_fmt_percent_pm(deg_m, deg_s)}")

    if show_debug_time_deltas:
        med_m, med_s = get_pm("median_delta_time_s")
        mean_m, mean_s = get_pm("mean_delta_time_s")
        print(
            f"  [debug] Median delta time:            {_fmt_time_pm(med_m, med_s)}  (agent - baseline)"
        )
        print(
            f"  [debug] Mean delta time:              {_fmt_time_pm(mean_m, mean_s)}  (agent - baseline)"
        )


def print_table_a_runtime_both_modes(
    table_a_time: Dict[str, Any],
    include_baseline_sanity: bool = False,
    show_per_seed: bool = True,
    show_debug_time_deltas: bool = False,
    show_mean_pm_block: bool = False,
) -> None:
    seeds = table_a_time["seeds"]
    pooled = table_a_time["pooled"]
    per_seed = table_a_time["per_seed"]
    agg = table_a_time["by_seed_agg"]

    print(
        "----Table A (RUNTIME MAIN, percent-based; POOLED over all test instances)----"
    )
    print(
        f"Seeds used: {seeds} | n_seeds={table_a_time['n_seeds']} | n_test_total={table_a_time['n_test_total']}"
    )

    _print_block_pooled_time(
        "Agent vs GrevLex (time)", "agent_vs_grevlex", pooled, show_debug_time_deltas
    )
    _print_block_pooled_time(
        "Agent vs DegLex (time)", "agent_vs_deglex", pooled, show_debug_time_deltas
    )
    if include_baseline_sanity:
        _print_block_pooled_time(
            "DegLex vs GrevLex (time; sanity check)",
            "deglex_vs_grevlex",
            pooled,
            show_debug_time_deltas,
        )

    print()
    print("  [pooled worst loss outliers]")
    for pref in ["agent_vs_grevlex", "agent_vs_deglex"]:
        outs = table_a_time.get("pooled_outliers", {}).get(pref, [])
        if not outs:
            print(f"    {pref}: (no losses / no outliers)")
            continue
        print(f"    {pref}: top loss ratios (agent/base)")
        for o in outs[:5]:
            print(
                f"      ratio={o['ratio']:.3g}  agent={o['agent_time_s']:.6g}s  base={o['base_time_s']:.6g}s  "
                f"delta={o['delta_s']:.3g}s  pct={o['pct_slowdown']:.3g}%"
            )
    print()

    print("----Table A (RUNTIME APPENDIX; median [IQR] across seeds)----")
    print(f"Seeds used: {seeds} | n_seeds={table_a_time['n_seeds']}")

    _print_block_by_seed_iqr_time(
        "Agent vs GrevLex (time)",
        "agent_vs_grevlex",
        per_seed,
        seeds,
        show_debug_time_deltas,
    )
    _print_block_by_seed_iqr_time(
        "Agent vs DegLex (time)",
        "agent_vs_deglex",
        per_seed,
        seeds,
        show_debug_time_deltas,
    )
    if include_baseline_sanity:
        _print_block_by_seed_iqr_time(
            "DegLex vs GrevLex (time; sanity check)",
            "deglex_vs_grevlex",
            per_seed,
            seeds,
            show_debug_time_deltas,
        )
    print()

    if show_mean_pm_block:
        print("----Table A (RUNTIME; mean +- std across seeds)----")
        _print_block_by_seed_time(
            "Agent vs GrevLex (time)", "agent_vs_grevlex", agg, show_debug_time_deltas
        )
        _print_block_by_seed_time(
            "Agent vs DegLex (time)", "agent_vs_deglex", agg, show_debug_time_deltas
        )
        if include_baseline_sanity:
            _print_block_by_seed_time(
                "DegLex vs GrevLex (time; sanity check)",
                "deglex_vs_grevlex",
                agg,
                show_debug_time_deltas,
            )
        print()

    if show_per_seed:
        print("Per-seed breakdown (runtime; percent-based):")
        for s in seeds:
            d = per_seed[s]
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
                    f"faster={win:.3f}%, tie={tie:.3f}%, slower={loss:.3f}%, "
                    f"speedup_faster_only={imp:.3f}%, slowdown_slower_only={deg:.3f}%"
                )

                lr99 = d.get(pref + "_loss_ratio_p99", float("nan"))
                lrmax = d.get(pref + "_loss_ratio_max", float("nan"))
                bmin = d.get(pref + "_loss_base_time_min_s", float("nan"))
                ge50 = d.get(pref + "_loss_ratio_ge_50_percent", float("nan"))
                msg += f" | loss_ratio_p99={lr99:.3g}, loss_ratio_max={lrmax:.3g}, loss_base_min={bmin:.3g}s, loss_%>=50x={ge50:.3g}%"

                if show_debug_time_deltas:
                    med = d.get(pref + "_median_delta_time_s", float("nan"))
                    mean = d.get(pref + "_mean_delta_time_s", float("nan"))
                    msg += f", [debug] med_delta={med:.6g} (s), mean_delta={mean:.6g} (s)  (agent-baseline)"

                outs = (
                    table_a_time.get("outliers_per_seed", {}).get(s, {}).get(pref, [])
                )
                if outs:
                    o = outs[0]
                    msg += f" | worst_ratio={o['ratio']:.3g}(agent={o['agent_time_s']:.3g}s, base={o['base_time_s']:.3g}s)"

                print(msg)
        print()
