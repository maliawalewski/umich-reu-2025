from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rcparams():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["FreeSerif", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "font.weight": "normal",
            "axes.labelweight": "normal",
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def format_axes(ax):
    ax.grid(True, linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)


def _safe_pos(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    a = a[a > 0]
    return a


def _collect_runtime_ratio_per_seed(
    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]],
    *,
    baseline: Literal["grevlex", "deglex"] = "grevlex",
) -> Dict[int, np.ndarray]:
    """
    Returns {seed: ratios} with ratios = agent_time_s / baseline_time_s.
    """
    out: Dict[int, np.ndarray] = {}
    for seed, kinds in sorted(dfs_by_seed.items()):
        if "test_metrics" not in kinds:
            continue
        df = kinds["test_metrics"]

        need = {"agent_time_s", "grevlex_time_s", "deglex_time_s"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"seed {seed}: test_metrics missing columns: {missing}")

        agent_t = df["agent_time_s"].to_numpy(dtype=float)
        base_t = (
            df["grevlex_time_s"].to_numpy(dtype=float)
            if baseline == "grevlex"
            else df["deglex_time_s"].to_numpy(dtype=float)
        )

        ok = np.isfinite(agent_t) & np.isfinite(base_t) & (agent_t > 0) & (base_t > 0)
        r = (agent_t[ok] / base_t[ok]).astype(float)
        r = r[np.isfinite(r) & (r > 0)]
        if r.size:
            out[int(seed)] = r
    if not out:
        raise RuntimeError("No runtime ratios found to plot (no valid test_metrics?).")
    return out


def _validate_clip(q: Tuple[float, float]) -> Tuple[float, float]:
    qlo, qhi = float(q[0]), float(q[1])
    if not (0.0 <= qlo <= 1.0 and 0.0 <= qhi <= 1.0):
        raise ValueError(f"Quantiles must be in [0,1]. Got {q}")
    if qlo >= qhi:
        raise ValueError(f"Need qlo < qhi. Got {q}")
    return qlo, qhi


def _apply_quantile_clip(x: np.ndarray, q: Tuple[float, float]) -> np.ndarray:
    qlo, qhi = _validate_clip(q)
    lo, hi = np.quantile(x, [qlo, qhi])
    return x[(x >= lo) & (x <= hi)]


def _cactus_xy(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cactus plot coordinates:
      - sort x
      - x-axis is percentile (0..100)
      - y-axis is value
    """
    xs = np.sort(np.asarray(x, dtype=float))
    n = xs.size
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    px = np.linspace(0.0, 100.0, n, dtype=float)
    return px, xs


def plot_runtime_ratio_cactus(
    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]],
    outpath: Path,
    *,
    baseline: Literal["grevlex", "deglex"] = "grevlex",
    baseline_name: str = "DegRevLex",
    title: Optional[str] = None,
    figsize=(3.25, 2.35),
    legend: bool = True,
    logy: bool = False,
    y_clip_quantiles: Optional[Tuple[float, float]] = None,  # e.g. (0.01, 0.99)
    per_seed_overlay: bool = False,
) -> None:
    """
    Cactus plot of runtime ratio r = agent_time / baseline_time.
    Sort ratios and plot value vs percentile rank.

    - r < 1.0 => agent faster
    - r = 1.0 => tie
    - r > 1.0 => agent slower
    """
    rcparams()

    per_seed = _collect_runtime_ratio_per_seed(dfs_by_seed, baseline=baseline)
    pooled = np.concatenate(list(per_seed.values()), axis=0)

    pooled_plot = pooled
    if y_clip_quantiles is not None:
        pooled_plot = _apply_quantile_clip(pooled_plot, y_clip_quantiles)

    fig, ax = plt.subplots(figsize=figsize)

    if per_seed_overlay:
        for _, r in per_seed.items():
            r_plot = r
            if y_clip_quantiles is not None and r_plot.size:
                r_plot = _apply_quantile_clip(r_plot, y_clip_quantiles)
            px_s, ys_s = _cactus_xy(r_plot)
            if ys_s.size:
                ax.plot(px_s, ys_s, linewidth=0.8, alpha=0.3)

    px, ys = _cactus_xy(pooled_plot)

    med = float(np.median(pooled)) if pooled.size else float("nan")

    px, ys = _cactus_xy(pooled_plot)

    med = float(np.median(pooled)) if pooled.size else float("nan")
    p90 = float(np.quantile(pooled, 0.90)) if pooled.size else float("nan")
    p99 = float(np.quantile(pooled, 0.99)) if pooled.size else float("nan")

    print(
        f"[runtime_cactus] {outpath.name} | baseline={baseline_name} | "
        f"n={pooled.size} | median={med:.6g} | p90={p90:.6g} | p99={p99:.6g}"
    )
    if y_clip_quantiles is not None and pooled_plot.size:
        qlo, qhi = y_clip_quantiles
        print(
            f"[runtime_cactus]   (plotted with y-clip quantiles: {qlo:g}..{qhi:g}; "
            f"kept {pooled_plot.size}/{pooled.size})"
        )

    label = f"Agent / {baseline_name}"
    if np.isfinite(med):
        label += f" (median {med:.3g})"

    ax.plot(px, ys, linewidth=1.8, label=label)

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
        color="black",
        label="1.0 (tie)" if legend else "_nolegend_",
    )

    ax.set_xlabel("Percentile of test instances (sorted by runtime ratio)")
    ax.set_ylabel(f"Runtime ratio (agent / {baseline_name})")

    if logy:
        ax.set_yscale("log")

    if title:
        ax.set_title(title)

    format_axes(ax)

    if legend:
        ax.legend(frameon=True, fontsize=7, loc="upper left")

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def make_runtime_cactus_figs(
    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]],
    outdir: Path,
    *,
    baseset: str,
    make_appendix_deglex: bool = True,
    logy: bool = False,
    y_clip_quantiles: Optional[Tuple[float, float]] = None,
    per_seed_overlay: bool = False,
) -> None:
    """
    Writes:
      - runtime_cactus_{baseset}_vs_degrevlex.pdf  (main)
      - runtime_cactus_{baseset}_vs_deglex.pdf    (appendix)
    """
    outdir.mkdir(parents=True, exist_ok=True)

    plot_runtime_ratio_cactus(
        dfs_by_seed,
        outdir / f"runtime_cactus_{baseset}_vs_degrevlex.pdf",
        baseline="grevlex",
        baseline_name="GrevLex",
        title=None,
        legend=True,
        logy=logy,
        y_clip_quantiles=y_clip_quantiles,
        per_seed_overlay=per_seed_overlay,
    )

    if make_appendix_deglex:
        plot_runtime_ratio_cactus(
            dfs_by_seed,
            outdir / f"runtime_cactus_{baseset}_vs_deglex.pdf",
            baseline="deglex",
            baseline_name="GrLex",
            title=None,
            legend=True,
            logy=logy,
            y_clip_quantiles=y_clip_quantiles,
            per_seed_overlay=per_seed_overlay,
        )
