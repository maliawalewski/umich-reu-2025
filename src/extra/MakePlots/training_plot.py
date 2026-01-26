from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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


def robust_limits(a: np.ndarray, lo=5.0, hi=95.0, pad=0.10):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (-1.0, 1.0)
    lo_v, hi_v = np.percentile(a, [lo, hi])
    span = hi_v - lo_v
    if span <= 0:
        span = 1.0
    return lo_v - pad * span, hi_v + pad * span


def format_axes(ax):
    ax.grid(True, linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)


def _rolling_mean(y: pd.Series, window: int) -> pd.Series:
    window = int(max(1, window))
    return y.rolling(window=window, min_periods=1).mean()


def _prep_agent_series(df_agent: pd.DataFrame, x: str) -> pd.DataFrame:
    need = {x, "raw_reward", "delta_vs_grevlex_reward"}
    missing = [c for c in need if c not in df_agent.columns]
    if missing:
        raise ValueError(f"train_agent_metrics missing columns: {missing}")

    g = (
        df_agent[[x, "raw_reward", "delta_vs_grevlex_reward"]]
        .groupby(x, as_index=False)
        .mean(numeric_only=True)
    )
    g = g.sort_values(x)
    return g


def _prep_baseline_series(df_base: pd.DataFrame, x: str) -> pd.DataFrame:
    need = {x, "grevlex_mean_reward", "deglex_mean_reward"}
    missing = [c for c in need if c not in df_base.columns]
    if missing:
        raise ValueError(f"train_baseline_metrics missing columns: {missing}")

    g = df_base[[x, "grevlex_mean_reward", "deglex_mean_reward"]].copy()
    g["deglex_minus_grevlex_mean_reward"] = (
        g["deglex_mean_reward"] - g["grevlex_mean_reward"]
    )
    g = g.sort_values(x)
    return g


def make_training_plot(
    dfs_by_seed: Dict[int, Dict[str, pd.DataFrame]],
    outpath: Path,
    *,
    mode: Literal["raw", "delta"] = "raw",
    xaxis: Literal["episode", "global_timestep"] = "episode",
    window: int = 400,
    include_deglex_reference: bool = True,
    title: Optional[str] = None,
    figsize=(3.25, 2.35),
    band: Literal["iqr", "std", "none"] = "iqr",
    legend: bool = True,
    inset: bool = False,
    inset_ylim: Optional[tuple[float, float]] = None,
) -> None:
    """
    mode="raw":
        - per-seed smoothed agent raw_reward (thin)
        - bold mean across seeds
        - dashed smoothed GrevLex and DegLex baselines (mean across seeds)

    mode="delta":
        - per-seed smoothed agent delta_vs_grevlex_reward (thin)
        - bold mean across seeds
        - dashed y=0 GrevLex line
        - optional dashed DegLex delta line from train_baseline_metrics mean( deglex - grevlex )

    xaxis chooses which x to group on. Recommended:
        - episode (best if you want baselines aligned cleanly)
    """
    rcparams()

    per_seed = {}
    for seed, kinds in sorted(dfs_by_seed.items()):
        if "train_agent_metrics" not in kinds or "train_baseline_metrics" not in kinds:
            continue

        agent = _prep_agent_series(kinds["train_agent_metrics"], xaxis)
        base = _prep_baseline_series(kinds["train_baseline_metrics"], "episode")

        if xaxis == "episode":
            merged = agent.merge(base, on="episode", how="left")
        else:
            if "episode" not in kinds["train_agent_metrics"].columns:
                raise ValueError(
                    "train_agent_metrics missing 'episode' needed to align baselines for global_timestep x-axis"
                )
            agent_with_ep = kinds["train_agent_metrics"][
                ["global_timestep", "episode", "raw_reward", "delta_vs_grevlex_reward"]
            ]
            agent_with_ep = agent_with_ep.groupby(
                ["global_timestep", "episode"], as_index=False
            ).mean(numeric_only=True)
            merged = agent_with_ep.merge(base, on="episode", how="left").sort_values(
                "global_timestep"
            )

        merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[xaxis, "raw_reward", "delta_vs_grevlex_reward"]
        )
        per_seed[seed] = merged

    if not per_seed:
        raise RuntimeError(
            "No seeds had both train_agent_metrics and train_baseline_metrics."
        )

    y_col = "raw_reward" if mode == "raw" else "delta_vs_grevlex_reward"

    wide_agent = []
    wide_grev = []
    wide_deg = []
    wide_deg_delta = []

    for seed, df in per_seed.items():
        x = df[xaxis].to_numpy()
        wide_agent.append(
            pd.DataFrame({xaxis: x, f"seed_{seed}": df[y_col].to_numpy()})
        )

        wide_grev.append(
            pd.DataFrame(
                {xaxis: x, f"seed_{seed}": df["grevlex_mean_reward"].to_numpy()}
            )
        )
        wide_deg.append(
            pd.DataFrame(
                {xaxis: x, f"seed_{seed}": df["deglex_mean_reward"].to_numpy()}
            )
        )
        wide_deg_delta.append(
            pd.DataFrame(
                {
                    xaxis: x,
                    f"seed_{seed}": df["deglex_minus_grevlex_mean_reward"].to_numpy(),
                }
            )
        )

    def outer_merge(frames):
        out = frames[0]
        for f in frames[1:]:
            out = out.merge(f, on=xaxis, how="outer")
        out = out.sort_values(xaxis)
        return out

    A = outer_merge(wide_agent)
    G = outer_merge(wide_grev)
    D = outer_merge(wide_deg)
    DD = outer_merge(wide_deg_delta)

    x_vals = A[xaxis].to_numpy()

    seed_cols = [c for c in A.columns if c != xaxis]
    agent_seed_smoothed = {}
    for c in seed_cols:
        s = pd.Series(A[c].to_numpy())
        agent_seed_smoothed[c] = _rolling_mean(s, window=window).to_numpy()

    agent_mean = np.nanmean(
        np.vstack([agent_seed_smoothed[c] for c in seed_cols]), axis=0
    )

    agent_stack = np.vstack([agent_seed_smoothed[c] for c in seed_cols])

    if band == "iqr":
        agent_lo = np.nanpercentile(agent_stack, 25, axis=0)
        agent_hi = np.nanpercentile(agent_stack, 75, axis=0)
        band_label = "IQR"
    elif band == "std":
        agent_std = np.nanstd(agent_stack, axis=0)
        agent_lo = agent_mean - agent_std
        agent_hi = agent_mean + agent_std
        band_label = "±1 std"
    else:
        agent_lo = agent_hi = None
        band_label = None

    grev_mean = np.nanmean(G[seed_cols].to_numpy(dtype=float), axis=1)
    deg_mean = np.nanmean(D[seed_cols].to_numpy(dtype=float), axis=1)
    deg_delta_mean = np.nanmean(DD[seed_cols].to_numpy(dtype=float), axis=1)

    grev_mean_s = _rolling_mean(pd.Series(grev_mean), window=window).to_numpy()
    deg_mean_s = _rolling_mean(pd.Series(deg_mean), window=window).to_numpy()
    deg_delta_mean_s = _rolling_mean(
        pd.Series(deg_delta_mean), window=window
    ).to_numpy()

    deg_delta_const = float(np.nanmean(deg_delta_mean))

    fig, ax = plt.subplots(figsize=figsize)

    if band != "none":
        ax.fill_between(
            x_vals,
            agent_lo,
            agent_hi,
            alpha=0.40,
            linewidth=0.0,
            label=f"Agent ({band_label})",
        )

    ax.plot(x_vals, agent_mean, linewidth=1.8, label="Agent")

    if mode == "raw":
        ax.plot(
            x_vals,
            grev_mean_s,
            linestyle="--",
            linewidth=1.0,
            label="GrevLex",
            color="black",
        )
        if include_deglex_reference:
            ax.plot(
                x_vals,
                deg_mean_s,
                linestyle="--",
                linewidth=1.0,
                label="GrLex",
                color="orange",
            )
        ax.set_ylabel("Reward (no baseline)")
    else:
        ax.plot(
            [x_vals[0], x_vals[-1]],
            [0.0, 0.0],
            linestyle=":",
            linewidth=0.9,
            color="black",
            label="GrevLex" if legend else "_nolegend_",
        )

        if include_deglex_reference:
            ax.plot(
                [x_vals[0], x_vals[-1]],
                [deg_delta_const, deg_delta_const],
                linestyle="--",
                linewidth=1.0,
                label="GrLex",
                color="orange",
            )
        ax.set_ylabel("Reward delta vs GrevLex")

    ax.set_xlabel("Episode" if xaxis == "episode" else "Global timestep")

    if title:
        ax.set_title(title)

    if inset:
        axins = inset_axes(ax, width="33%", height="33%", loc="best")
        axins.set_xticks([])
        axins.set_yticks([])
        axins.grid(False)

        frac = 0.60
        x0 = x_vals[int((1.0 - frac) * len(x_vals))]
        x1 = x_vals[-1]

        axins.plot(x_vals, agent_mean, linewidth=1.2)

        if mode == "raw":
            axins.plot(
                x_vals, grev_mean_s, linestyle="--", linewidth=0.9, color="black"
            )
            if include_deglex_reference:
                axins.plot(
                    x_vals, deg_mean_s, linestyle="--", linewidth=0.9, color="orange"
                )
        else:
            axins.plot(
                [x_vals[0], x_vals[-1]],
                [0.0, 0.0],
                linestyle="--",
                linewidth=0.9,
                color="black",
            )
            if include_deglex_reference:
                axins.plot(
                    [x_vals[0], x_vals[-1]],
                    [deg_delta_const, deg_delta_const],
                    linestyle="--",
                    linewidth=0.9,
                    color="orange",
                )

        axins.set_xlim(x0, x1)

        if mode == "delta":
            axins.set_ylim(-120, 20)

        axins.grid(True, linewidth=0.4, alpha=0.25)
        axins.tick_params(axis="both", labelsize=6)

    format_axes(ax)
    if legend:
        ax.legend(frameon=True, fontsize=7, loc="lower right")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
