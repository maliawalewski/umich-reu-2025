from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

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


def _collect_deltas(
    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]],
    *,
    baseline: str = "grevlex",  # or "deglex"
) -> np.ndarray:
    """
    Pool reward deltas across all seeds.

    baseline="grevlex": agent_reward - grevlex_reward
    baseline="deglex" : agent_reward - deglex_reward
    """
    baseline_reward_col = f"{baseline}_reward"
    all_d = []
    for seed, kinds in sorted(dfs_by_seed.items()):
        if "test_metrics" not in kinds:
            continue
        df = kinds["test_metrics"]

        need = {"agent_reward", baseline_reward_col}
        if not need.issubset(df.columns):
            raise ValueError(f"seed {seed}: test_metrics missing {need}")

        d = (df["agent_reward"] - df[baseline_reward_col]).to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        all_d.append(d)

    if not all_d:
        raise RuntimeError("No test_metrics found to collect deltas.")
    return np.concatenate(all_d, axis=0)


def plot_delta_pmf(
    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]],
    outpath: Path,
    *,
    round_to: Optional[float] = None,
    top_k: int = 6,
    min_prob: float = 0.005,  # 0.5% cutoff
    title: Optional[str] = None,
    figsize=(3.25, 2.35),
    baseline_name: str = "GrevLex",
    delta_fmt: str = ".4g",
    tie_tol: float = 0.0, 
    baseline_key: str = "grevlex",
) -> None:

    rcparams()
    deltas = _collect_deltas(dfs_by_seed, baseline=baseline_key)

    v = np.asarray(deltas, dtype=float)
    v = v[np.isfinite(v)]
    if round_to is not None and round_to > 0:
        v = np.round(v / round_to) * round_to

    s = pd.Series(v)
    counts = s.value_counts(dropna=True)
    n = float(counts.sum())
    probs = counts / n

    kept: list[tuple[float, float]] = []
    other_mass = 0.0
    for delta_val, p in probs.items():
        p = float(p)
        dv = float(delta_val)
        if (p >= min_prob) and (len(kept) < top_k):
            kept.append((dv, p))
        else:
            other_mass += p

    if len(kept) == 0 and len(probs) > 0:
        dv = float(probs.index[0])
        p = float(probs.iloc[0])
        kept.append((dv, p))
        other_mass = float(1.0 - p)

    kept.sort(key=lambda t: t[0])

    def fmt_delta(dv: float) -> str:
        if abs(dv) <= tie_tol:
            return "0 (tie)"
        return format(dv, delta_fmt)

    rows = [(fmt_delta(dv), p) for dv, p in kept]
    if other_mass > 0:
        rows.append(("Other", other_mass))

    labels = [lab for lab, _ in rows]
    masses = np.array([p for _, p in rows], dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(labels))
    bars = ax.barh(y, masses)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel("Probability mass")
    ax.set_ylabel(rf"Reward delta vs {baseline_name}")

    xmax = float(np.max(masses)) if masses.size else 1.0
    right = max(0.05, xmax * 1.18)
    ax.set_xlim(0.0, right)

    pad = 0.015 * right
    for rect, p in zip(bars, masses):
        x = rect.get_width()
        yc = rect.get_y() + rect.get_height() / 2.0
        txt = f"{100*p:.1f}%"

        if x + pad > 0.98 * right:
            ax.text(x - pad, yc, txt, va="center", ha="right", fontsize=7)
        else:
            ax.text(x + pad, yc, txt, va="center", ha="left", fontsize=7)

    if title:
        ax.set_title(title)

    format_axes(ax)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def make_test_delta_figs(
    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]],
    outdir: Path,
    *,
    baseset: str,
    round_to: Optional[float] = None,
) -> None:

    outdir.mkdir(parents=True, exist_ok=True)

    plot_delta_pmf(
        dfs_by_seed,
        outdir / f"delta_pmf_vs_degrevlex_{baseset}.pdf",
        round_to=round_to,
        top_k=6,
        min_prob=0.005,
        baseline_key="grevlex",
        baseline_name="GrevLex",
    )

    plot_delta_pmf(
        dfs_by_seed,
        outdir / f"delta_pmf_vs_deglex_{baseset}.pdf",
        round_to=round_to,
        top_k=6,
        min_prob=0.005,
        baseline_key="deglex",
        baseline_name="GrLex",
    )
