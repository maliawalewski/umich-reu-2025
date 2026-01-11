import argparse
import re
from pathlib import Path
from typing import Dict

import pandas as pd

from table_a_reward import compute_table_a_reward, print_table_a_both_modes
from weights_table import weights_table_from_dfs
from training_plot import make_training_plot

# Example:
#   python main.py --baseset TRIANGULATION_BASE_SET
#   python main.py --baseset TRIANGULATION_BASE_SET --show-per-seed
#   python main.py --baseset TRIANGULATION_BASE_SET --include-baseline-sanity

KIND_SUFFIXES = {
    "final_agent_weight_vector": "_final_agent_weight_vector.csv",
    "test_metrics": "_test_metrics.csv",
    "train_agent_metrics": "_train_agent_metrics.csv",
    "train_baseline_metrics": "_train_baseline_metrics.csv",
    "train_losses": "_train_losses.csv",
}

FILENAME_RE = re.compile(
    r"^td3_run_baseset_(?P<baseset>.+?)_seed_(?P<seed>\d+?)_(?P<kind>.+?)\.csv$"
)


def get_src_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def get_root_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def scan_results(results_dir: Path, baseset: str) -> Dict[int, Dict[str, Path]]:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results_dir does not exist: {results_dir}")

    grouped: Dict[int, Dict[str, Path]] = {}
    for p in results_dir.glob("td3_run_baseset_*_seed_*_*.csv"):
        m = FILENAME_RE.match(p.name)
        if not m:
            continue
        if m.group("baseset") != baseset:
            continue

        seed = int(m.group("seed"))
        kind = m.group("kind")
        if kind not in KIND_SUFFIXES:
            continue

        grouped.setdefault(seed, {})
        if kind in grouped[seed] and grouped[seed][kind] != p:
            raise RuntimeError(
                f"Duplicate kind for seed={seed}, kind={kind}:\n"
                f"  {grouped[seed][kind]}\n"
                f"  {p}"
            )
        grouped[seed][kind] = p

    return grouped


def load_grouped(
    grouped_paths: Dict[int, Dict[str, Path]],
) -> Dict[int, Dict[str, pd.DataFrame]]:
    out: Dict[int, Dict[str, pd.DataFrame]] = {}
    for seed, kinds in sorted(grouped_paths.items()):
        out[seed] = {}
        for kind, path in sorted(kinds.items()):
            out[seed][kind] = load_csv(path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseset", type=str, required=True, help="example: TRIANGULATION_BASE_SET"
    )
    ap.add_argument(
        "--src",
        type=str,
        default=None,
        help="Optional override for src/ directory (otherwise inferred from script location).",
    )
    ap.add_argument(
        "--require-all-kinds",
        action="store_true",
        help="If set, error unless every seed has all expected CSV kinds.",
    )
    ap.add_argument(
        "--quiet-scan",
        action="store_true",
        help="If set, suppress per-seed scan output (still prints Table A).",
    )
    ap.add_argument(
        "--include-baseline-sanity",
        action="store_true",
        help="If set, also print DegLex vs GrevLex sanity-check block.",
    )
    ap.add_argument(
        "--show-debug-reward-deltas",
        action="store_true",
        help="If set, print debug reward-unit deltas alongside the percent-based stats.",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="figures",
        help="Where to write plots (relative to src/ unless absolute).",
    )
    ap.add_argument(
        "--make-training-plot",
        action="store_true",
        help="If set, write the training curve plot PDF.",
    )
    ap.add_argument(
        "--training-mode",
        choices=["raw", "delta"],
        default="raw",
        help="raw=absolute rewards, delta=Δ vs GrevLex.",
    )
    ap.add_argument(
        "--training-xaxis",
        choices=["episode", "global_timestep"],
        default="episode",
        help="X-axis for training plot.",
    )
    ap.add_argument(
        "--training-window",
        type=int,
        default=400,
        help="Moving average window (in x-axis points).",
    )

    args = ap.parse_args()

    src_dir = Path(args.src).resolve() if args.src else get_src_dir()
    results_dir = (src_dir / "results").resolve()

    grouped_paths = scan_results(results_dir, args.baseset)
    if not grouped_paths:
        raise FileNotFoundError(
            f"No CSVs found for baseset={args.baseset} in {results_dir}"
        )

    expected_kinds = list(KIND_SUFFIXES.keys())
    seeds = sorted(grouped_paths.keys())

    print(f"src_dir     = {src_dir}")
    print(f"results_dir = {results_dir}")
    print(f"baseset     = {args.baseset}")
    print(f"seeds       = {seeds}")
    print()

    missing_any = False
    if not args.quiet_scan:
        for seed in seeds:
            kinds = grouped_paths[seed]
            missing = [k for k in expected_kinds if k not in kinds]
            present = [k for k in expected_kinds if k in kinds]
            print(f"[seed {seed}]")
            print(f"  present: {present}")
            if missing:
                missing_any = True
                print(f"  missing: {missing}")
            for k in present:
                print(f"    - {k}: {kinds[k].name}")
            print()

    if args.require_all_kinds and missing_any:
        raise RuntimeError(
            "Some seeds are missing one or more expected CSV kinds (see report above)."
        )

    dfs_by_seed = load_grouped(grouped_paths)

    if not args.quiet_scan:
        for seed in seeds:
            print(f"----Loaded seed {seed}----")
            for kind, df in dfs_by_seed[seed].items():
                print(f"[{kind}] shape={df.shape} cols={list(df.columns)}")
            print()

    table_a = compute_table_a_reward(dfs_by_seed)

    print_table_a_both_modes(
        table_a,
        include_baseline_sanity=args.include_baseline_sanity,
        show_per_seed=True,
        show_debug_reward_deltas=args.show_debug_reward_deltas,
    )

    weights_table_from_dfs(dfs_by_seed, action_scale=1e3, show_int=True)

    root_dir = get_root_dir()

    if args.make_training_plot:
        outdir = Path(args.outdir)
        if not outdir.is_absolute():
            outdir = (root_dir / outdir).resolve()

        outpath = (
            outdir
            / f"training_curve_{args.baseset}_{args.training_mode}_{args.training_xaxis}.pdf"
        )

        make_training_plot(
            dfs_by_seed,
            outpath,
            mode=args.training_mode,
            xaxis=args.training_xaxis,
            window=args.training_window,
            include_deglex_reference=True,
            title=None,
            band="iqr",
        )
        print(f"Wrote training plot: {outpath}")

    return dfs_by_seed


if __name__ == "__main__":
    main()
