import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pearsonr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / denom) if denom != 0 else float("nan")


def rankdata(a):
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    s = a[order]
    n = len(a)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def spearmanr(x, y):
    return pearsonr(rankdata(x), rankdata(y))


def rcparams():
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


def clean_df(df):
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["time_s", "reward", "baseline_reward", "reward_delta"]
    )
    df = df[df["time_s"] > 0].copy()
    return df


def format_axes(ax):
    ax.grid(True, linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)


def robust_limits(a, lo=0.5, hi=99.5, pad=0.06):
    lo_v, hi_v = np.percentile(a, [lo, hi])
    span = hi_v - lo_v
    if span <= 0:
        span = 1.0
    return lo_v - pad * span, hi_v + pad * span


def fit_linear_trend(x, y):
    m, b = np.polyfit(x, y, 1)
    return float(m), float(b)


def plot_scatter_with_trend(ax, x, y, add_trend=True):
    ax.scatter(x, y, s=10, alpha=0.30, linewidths=0, rasterized=True)

    if add_trend and len(x) >= 2:
        m, b = fit_linear_trend(x, y)
        xx = np.linspace(x.min(), x.max(), 200)
        yy = m * xx + b
        ax.plot(xx, yy) 


def make_main(df, outpath, add_trend=True, use_logx=False):
    x = df["time_s"].to_numpy(dtype=float)
    y = df["reward_delta"].to_numpy(dtype=float)

    x_for_corr = np.log10(x) if use_logx else x
    rp = pearsonr(x_for_corr, y)
    rs = spearmanr(x_for_corr, y)

    fig, ax = plt.subplots(figsize=(3.25, 2.35))

    if use_logx:
        ax.set_xscale("log")

    plot_scatter_with_trend(ax, x, y, add_trend=add_trend)

    ax.set_xlabel("Computation time (s)")
    ax.set_ylabel("Reward (GrevLex baseline)")

    ax.set_xlim(*robust_limits(x, lo=0.5, hi=99.5, pad=0.06))
    ax.set_ylim(*robust_limits(y, lo=0.5, hi=99.5, pad=0.08))

    format_axes(ax)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    return rp, rs, len(df)


def make_appendix(df, outpath, add_trend=True, use_logx=False):
    x = df["time_s"].to_numpy(dtype=float)
    y = df["reward"].to_numpy(dtype=float)
    baseline = float(df["baseline_reward"].iloc[0])

    x_for_corr = np.log10(x) if use_logx else x
    rp = pearsonr(x_for_corr, y)
    rs = spearmanr(x_for_corr, y)

    fig, ax = plt.subplots(figsize=(3.25, 2.35))

    if use_logx:
        ax.set_xscale("log")

    plot_scatter_with_trend(ax, x, y, add_trend=add_trend)

    ax.axhline(baseline, linestyle="--", linewidth=1.0)

    ax.set_xlabel("Computation time (s)")
    ax.set_ylabel("Reward (no baseline)")

    ax.set_xlim(*robust_limits(x, lo=0.5, hi=99.5, pad=0.06))
    ax.set_ylim(*robust_limits(y, lo=0.5, hi=99.5, pad=0.08))

    format_axes(ax)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    return rp, rs, len(df), baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="reward_vs_time.csv")
    parser.add_argument("--outdir", type=str, default="figures")
    parser.add_argument("--trend", action="store_true", help="add a linear trend line")
    parser.add_argument("--logx", action="store_true", help="use log scale on x-axis")
    args = parser.parse_args()

    rcparams()

    df = pd.read_csv(args.csv)
    df = clean_df(df)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    main_path = outdir / "reward_delta_vs_time_main_simple.pdf"
    app_path = outdir / "reward_vs_time_appendix_simple.pdf"

    rp, rs, n = make_main(df, main_path, add_trend=args.trend, use_logx=args.logx)
    arp, ars, an, baseline = make_appendix(df, app_path, add_trend=args.trend, use_logx=args.logx)

    corr_x = "log10(time)" if args.logx else "time"
    print(f"MAIN (reward_delta vs {corr_x})")
    print(f"N={n}, pearson={rp:.3f}, spearman={rs:.3f}")
    print()
    print(f"APPENDIX (reward absolute vs {corr_x})")
    print(f"N={an}, pearson={arp:.3f}, spearman={ars:.3f}, baseline_reward={baseline:.6g}")
    print()
    print(f"Wrote: {main_path}")
    print(f"Wrote: {app_path}")


if __name__ == "__main__":
    main()

