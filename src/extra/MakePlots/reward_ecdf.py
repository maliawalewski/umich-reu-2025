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


def _ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    xs = np.sort(x)
    ys = np.arange(1, xs.size + 1, dtype=float) / float(xs.size)
    return xs, ys


RewardMode = Literal["pct", "delta", "ratio"]


def _collect_reward_improvement(
    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]],
    *,
    baseline: Literal["grevlex", "deglex"] = "grevlex",
    mode: RewardMode = "pct",
    eps: float = 1e-12,
    denom_min: float = 1e-12,
) -> np.ndarray:
    """
    Collect a pooled 1D array across seeds from test_metrics:

    - mode="pct":  100 * (agent - base) / (abs(base) + eps)
    - mode="delta": (agent - base)
    - mode="ratio": agent / base   (filters abs(base) > denom_min)
    """
    all_x = []

    for seed, kinds in sorted(dfs_by_seed.items()):
        if "test_metrics" not in kinds:
            continue
        df = kinds["test_metrics"]

        need = {"agent_reward", "grevlex_reward", "deglex_reward"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"seed {seed}: test_metrics missing columns: {missing}")

        agent = df["agent_reward"].to_numpy(dtype=float)
        base = (
            df["grevlex_reward"].to_numpy(dtype=float)
            if baseline == "grevlex"
            else df["deglex_reward"].to_numpy(dtype=float)
        )

        ok = np.isfinite(agent) & np.isfinite(base)
        agent = agent[ok]
        base = base[ok]

        if agent.size == 0:
            continue

        if mode == "delta":
            x = (agent - base).astype(float)

        elif mode == "ratio":
            ok2 = np.abs(base) > denom_min
            x = (agent[ok2] / base[ok2]).astype(float)

        elif mode == "pct":
            denom = np.abs(base) + float(eps)
            ok2 = denom > denom_min
            x = (100.0 * (agent[ok2] - base[ok2]) / denom[ok2]).astype(float)

        else:
            raise ValueError(f"Unknown mode={mode!r}")

        x = x[np.isfinite(x)]
        if x.size:
            all_x.append(x)

    if not all_x:
        raise RuntimeError(
            "No reward improvements found to plot (no valid test_metrics?)."
        )
    return np.concatenate(all_x, axis=0)


def plot_reward_ecdf(
    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]],
    outpath: Path,
    *,
    baseline: Literal["grevlex", "deglex"] = "grevlex",
    baseline_name: str = "GrevLex",
    mode: RewardMode = "pct",
    title: Optional[str] = None,
    figsize=(3.25, 2.35),
    legend: bool = True,
    xscale: Literal["linear", "symlog"] = "linear",
    symlog_linthresh: float = 1.0,
    x_clip_quantiles: Optional[Tuple[float, float]] = None,  # e.g. (0.001, 0.999)
) -> None:
    """
    ECDF of reward improvement (pooled across seeds).

    For mode="pct":
      x = 100 * (agent_reward - baseline_reward) / (abs(baseline_reward) + eps)

      - x > 0 => agent better than baseline
      - x = 0 => tie
      - x < 0 => agent worse
    """
    rcparams()

    x = _collect_reward_improvement(dfs_by_seed, baseline=baseline, mode=mode)

    if x_clip_quantiles is not None:
        qlo, qhi = x_clip_quantiles
        lo, hi = np.quantile(x, [qlo, qhi])
        x_plot = x[(x >= lo) & (x <= hi)]
    else:
        x_plot = x

    xs, ys = _ecdf(x_plot)

    fig, ax = plt.subplots(figsize=figsize)

    med = float(np.median(x)) if x.size else float("nan")
    frac_pos = float(np.mean(x > 0.0)) if x.size else float("nan")

    if mode == "pct":
        xlabel = f"Reward % improvement vs {baseline_name}"
        tie_x = 0.0
        tie_label = "0.0 (tie)"
    elif mode == "delta":
        xlabel = f"Reward delta (agent - {baseline_name})"
        tie_x = 0.0
        tie_label = "0.0 (tie)"
    else:  # ratio
        xlabel = f"Reward ratio (agent / {baseline_name})"
        tie_x = 1.0
        tie_label = "1.0 (tie)"

    label = "Agent"
    if np.isfinite(med):
        if mode == "pct":
            label += f" (median {med:.3g}%, P[>0]={frac_pos:.3g})"
        else:
            label += f" (median {med:.3g}, P[>tie]={frac_pos:.3g})"

    ax.step(xs, ys, where="post", linewidth=1.8, label=label)

    ax.axvline(
        tie_x,
        linestyle="--",
        linewidth=1.0,
        color="black",
        label=tie_label if legend else "_nolegend_",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("ECDF (fraction of instances)")
    ax.set_ylim(0.0, 1.0)

    if xscale == "symlog":
        ax.set_xscale("symlog", linthresh=symlog_linthresh)

    if title:
        ax.set_title(title)

    format_axes(ax)
    if legend:
        ax.legend(frameon=True, fontsize=7, loc="lower right")

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def make_reward_ecdf_figs(
    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]],
    outdir: Path,
    *,
    baseset: str,
    mode: RewardMode = "pct",
    make_appendix_deglex: bool = True,
    xscale: Literal["linear", "symlog"] = "linear",
    symlog_linthresh: float = 1.0,
    x_clip_quantiles: Optional[Tuple[float, float]] = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    plot_reward_ecdf(
        dfs_by_seed,
        outdir / f"reward_ecdf_{baseset}_vs_grevlex_{mode}.pdf",
        baseline="grevlex",
        baseline_name="GrevLex",
        mode=mode,
        title=None,
        legend=True,
        xscale=xscale,
        symlog_linthresh=symlog_linthresh,
        x_clip_quantiles=x_clip_quantiles,
    )

    if make_appendix_deglex:
        plot_reward_ecdf(
            dfs_by_seed,
            outdir / f"reward_ecdf_{baseset}_vs_deglex_{mode}.pdf",
            baseline="deglex",
            baseline_name="DegLex",
            mode=mode,
            title=None,
            legend=True,
            xscale=xscale,
            symlog_linthresh=symlog_linthresh,
            x_clip_quantiles=x_clip_quantiles,
        )
