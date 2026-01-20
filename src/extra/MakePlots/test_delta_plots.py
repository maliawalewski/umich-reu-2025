from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, List

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
    all_d: List[np.ndarray] = []
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
    top_k: int = 15,
    min_prob: Optional[float] = None,  # set None to disable hard cutoff
    title: Optional[str] = None,
    figsize=(3.25, 2.35),
    baseline_name: str = "GrevLex",
    delta_fmt: str = ".4g",
    tie_tol: float = 0.0,
    baseline_key: str = "grevlex",
    # new behavior: keep largest-mass atoms up to top_k, or until target_mass reached (whichever first)
    target_mass: float = 0.90,
    show_other_unique: bool = True,
) -> None:
    """
    PMF-style barplot for reward deltas.

    Changes vs your original:
      1) tie_tol now actually buckets near-ties into exactly 0.0 (affects PMF, not just labels).
      2) selection is by descending probability (not iteration order), with optional cumulative-mass target.
      3) min_prob no longer silently causes "Other" to dominate; you can disable it (None) or keep as a guardrail.
      4) "Other" label can include how many distinct values it hides.
    """
    rcparams()
    deltas = _collect_deltas(dfs_by_seed, baseline=baseline_key)

    v = np.asarray(deltas, dtype=float)
    v = v[np.isfinite(v)]

    # (1) collapse near-ties into a single atom at 0.0
    if tie_tol is not None and tie_tol > 0:
        v[np.abs(v) <= tie_tol] = 0.0

    # optional rounding/bucketing
    if round_to is not None and round_to > 0:
        v = np.round(v / round_to) * round_to

    if v.size == 0:
        raise RuntimeError("No finite deltas to plot.")

    s = pd.Series(v)
    counts = s.value_counts(dropna=True)
    n = float(counts.sum())
    probs = (counts / n).sort_values(ascending=False)

    # (2) keep by mass, not by arbitrary iteration order
    kept: List[Tuple[float, float]] = []
    cum = 0.0
    for dv, p in probs.items():
        p = float(p)
        dv = float(dv)

        # optional hard cutoff if you still want it
        if (min_prob is not None) and (p < float(min_prob)):
            continue

        kept.append((dv, p))
        cum += p
        if len(kept) >= int(top_k) or cum >= float(target_mass):
            break

    # ensure at least one item is shown
    if len(kept) == 0 and len(probs) > 0:
        dv0 = float(probs.index[0])
        p0 = float(probs.iloc[0])
        kept = [(dv0, p0)]
        cum = p0

    other_mass = float(max(0.0, 1.0 - cum))

    # sort by delta value for nicer left-to-right interpretation on y-axis
    kept.sort(key=lambda t: t[0])

    def fmt_delta(dv: float) -> str:
        # Note: after bucketing, near-ties are literally 0.0 (if tie_tol>0),
        # so this mostly just makes the label friendly.
        if tie_tol is not None and tie_tol > 0 and abs(dv) <= tie_tol:
            return "0 (tie)"
        return format(dv, delta_fmt)

    rows: List[Tuple[str, float]] = [(fmt_delta(dv), p) for dv, p in kept]

    if other_mass > 0:
        if show_other_unique:
            other_unique = int(max(0, len(probs) - len(kept)))
            rows.append((f"Other ({other_unique} values)", other_mass))
        else:
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

    # You can tune tie_tol here if you want; 0.0 keeps old behavior.
    # A common choice if you have near-float noise is tie_tol = round_to (when provided).
    tie_tol = float(round_to) if (round_to is not None and round_to > 0) else 0.0

    plot_delta_pmf(
        dfs_by_seed,
        outdir / f"delta_pmf_vs_degrevlex_{baseset}.pdf",
        round_to=round_to,
        top_k=15,
        min_prob=None,  # disable hard cutoff so "Other" doesn't eat mass unfairly
        target_mass=0.90,  # show enough atoms to explain 90% of mass (capped by top_k)
        tie_tol=tie_tol,  # actually bucket near-ties
        baseline_key="grevlex",
        baseline_name="GrevLex",
        show_other_unique=True,
    )

    plot_delta_pmf(
        dfs_by_seed,
        outdir / f"delta_pmf_vs_deglex_{baseset}.pdf",
        round_to=round_to,
        top_k=15,
        min_prob=None,
        target_mass=0.90,
        tie_tol=tie_tol,
        baseline_key="deglex",
        baseline_name="GrLex",
        show_other_unique=True,
    )
