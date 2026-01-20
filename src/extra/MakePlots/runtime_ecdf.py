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


def _collect_runtime_ratio(
    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]],
    *,
    baseline: Literal["grevlex", "deglex"] = "grevlex",
) -> np.ndarray:
    all_r = []

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
            all_r.append(r)

    if not all_r:
        raise RuntimeError("No runtime ratios found to plot (no valid test_metrics?).")
    return np.concatenate(all_r, axis=0)


def _ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    xs = np.sort(x)
    ys = np.arange(1, xs.size + 1, dtype=float) / float(xs.size)
    return xs, ys


def plot_runtime_ecdf(
    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]],
    outpath: Path,
    *,
    baseline: Literal["grevlex", "deglex"] = "grevlex",
    baseline_name: str = "DegRevLex",
    title: Optional[str] = None,
    figsize=(3.25, 2.35),
    legend: bool = True,
    logx: bool = False,
    x_clip_quantiles: Optional[Tuple[float, float]] = None,  # e.g. (0.001, 0.999)
) -> None:
    """
    ECDF of runtime ratio r = agent_time / baseline_time (pooled across seeds).

    - r < 1.0 => agent faster
    - r = 1.0 => tie
    - r > 1.0 => agent slower
    """
    rcparams()

    r = _collect_runtime_ratio(dfs_by_seed, baseline=baseline)

    if x_clip_quantiles is not None:
        qlo, qhi = x_clip_quantiles
        lo, hi = np.quantile(r, [qlo, qhi])
        r_plot = r[(r >= lo) & (r <= hi)]
    else:
        r_plot = r

    xs, ys = _ecdf(r_plot)

    fig, ax = plt.subplots(figsize=figsize)

    med = float(np.median(r)) if r.size else float("nan")
    label = f"Agent / {baseline_name}"
    if np.isfinite(med):
        label += f" (median {med:.3g})"

    ax.step(xs, ys, where="post", linewidth=1.8, label=label)

    ax.axvline(
        1.0,
        linestyle="--",
        linewidth=1.0,
        color="black",
        label="1.0 (tie)" if legend else "_nolegend_",
    )

    ax.set_xlabel(f"Runtime ratio (agent / {baseline_name})")
    ax.set_ylabel("ECDF (fraction of instances)")

    if logx:
        ax.set_xscale("log")

    ax.set_ylim(0.0, 1.0)

    if title:
        ax.set_title(title)

    format_axes(ax)
    if legend:
        ax.legend(frameon=True, fontsize=7, loc="lower right")

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def make_runtime_ecdf_figs(
    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]],
    outdir: Path,
    *,
    baseset: str,
    main_baseline: Literal["grevlex", "deglex"] = "grevlex",
    make_appendix_deglex: bool = True,
    logx: bool = False,
    x_clip_quantiles: Optional[Tuple[float, float]] = None,
) -> None:

    outdir.mkdir(parents=True, exist_ok=True)

    plot_runtime_ecdf(
        dfs_by_seed,
        outdir / f"runtime_ecdf_{baseset}_vs_degrevlex.pdf",
        baseline="grevlex",
        baseline_name="GrevLex",
        title=None,
        legend=True,
        logx=logx,
        x_clip_quantiles=x_clip_quantiles,
    )

    if make_appendix_deglex:
        plot_runtime_ecdf(
            dfs_by_seed,
            outdir / f"runtime_ecdf_{baseset}_vs_deglex.pdf",
            baseline="deglex",
            baseline_name="GrLex",
            title=None,
            legend=True,
            logx=logx,
            x_clip_quantiles=x_clip_quantiles,
        )
