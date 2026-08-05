import argparse
import re
from pathlib import Path
from typing import Dict

import pandas as pd

from table_a_reward import compute_table_a_reward, print_table_a_both_modes
from weights_table import weights_table_from_dfs
from training_plot import make_training_plot
from test_delta_plots import make_test_delta_figs
from reward_ecdf import make_reward_ecdf_figs

KIND_SUFFIXES = {
    "final_agent_weight_vector": "_final_agent_weight_vector.csv",
    "test_metrics": "_test_metrics.csv",
    "train_agent_metrics": "_train_agent_metrics.csv",
    "train_baseline_metrics": "_train_baseline_metrics.csv",
    "train_losses": "_train_losses.csv",
}

FILENAME_RE = re.compile(
    r"^(?P<method>[a-z0-9]+)_run_baseset_(?P<baseset>.+?)_seed_(?P<seed>\d+?)_(?P<kind>.+?)\.csv$"
)


def get_src_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def get_root_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def scan_results(results_dir: Path, baseset: str) -> Dict[str, Dict[int, Dict[str, Path]]]:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results_dir does not exist: {results_dir}")

    grouped: Dict[str, Dict[int, Dict[str, Path]]] = {}
    for p in results_dir.glob("*_run_baseset_*_seed_*_*.csv"):
        m = FILENAME_RE.match(p.name)
        if not m:
            continue
        if m.group("baseset") != baseset:
            continue

        method = m.group("method")
        seed = int(m.group("seed"))
        kind = m.group("kind")
        if kind not in KIND_SUFFIXES:
            continue

        grouped.setdefault(method, {})
        grouped[method].setdefault(seed, {})
        if kind in grouped[method][seed] and grouped[method][seed][kind] != p:
            raise RuntimeError(
                f"Duplicate kind for method={method}, seed={seed}, kind={kind}:\n"
                f"  {grouped[method][seed][kind]}\n"
                f"  {p}"
            )
        grouped[method][seed][kind] = p

    return grouped


def load_grouped(
    grouped_paths: Dict[str, Dict[int, Dict[str, Path]]],
) -> Dict[str, Dict[int, Dict[str, pd.DataFrame]]]:
    out: Dict[str, Dict[int, Dict[str, pd.DataFrame]]] = {}
    for method, seeds_dict in sorted(grouped_paths.items()):
        out[method] = {}
        for seed, kinds in sorted(seeds_dict.items()):
            out[method][seed] = {}
            for kind, path in sorted(kinds.items()):
                out[method][seed][kind] = load_csv(path)
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
    ap.add_argument("--make-test-delta-plots", action="store_true")
    ap.add_argument(
        "--delta-round",
        type=float,
        default=None,
        help="optional rounding for delta plots, e.g. 1e-6",
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
    methods = sorted(grouped_paths.keys())

    print(f"src_dir     = {src_dir}")
    print(f"results_dir = {results_dir}")
    print(f"baseset     = {args.baseset}")
    print(f"methods     = {methods}")
    print()

    missing_any = False
    if not args.quiet_scan:
        for method in methods:
            for seed in sorted(grouped_paths[method].keys()):
                kinds = grouped_paths[method][seed]
                missing = [k for k in expected_kinds if k not in kinds]
                present = [k for k in expected_kinds if k in kinds]
                print(f"[{method} | seed {seed}]")
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

    dfs_by_method = load_grouped(grouped_paths)

    if not args.quiet_scan:
        for method in methods:
            for seed in sorted(dfs_by_method[method].keys()):
                print(f"----Loaded {method} seed {seed}----")
                for kind, df in dfs_by_method[method][seed].items():
                    print(f"[{kind}] shape={df.shape} cols={list(df.columns)}")
                print()

    # The rest of the plotting scripts currently assume td3 only. We extract td3
    # and pass it to them, or we could update the plotting scripts to handle dict of dicts.
    # For now, let's just pass td3 to the existing functions to not break everything.
    
    if "td3" not in dfs_by_method:
        print("No 'td3' data found, skipping table_a, weights_table, etc. (we need to update those to support other methods soon).")
        td3_dfs = {}
    else:
        td3_dfs = dfs_by_method["td3"]
        table_a = compute_table_a_reward(td3_dfs)

        print_table_a_both_modes(
            table_a,
            include_baseline_sanity=args.include_baseline_sanity,
            show_per_seed=True,
            show_debug_reward_deltas=args.show_debug_reward_deltas,
        )

        weights_table_from_dfs(td3_dfs, action_scale=1e3, show_int=True)

    root_dir = get_root_dir()

    if args.make_training_plot:
        outdir = Path(args.outdir)
        if not outdir.is_absolute():
            outdir = (root_dir / outdir).resolve()

        outpath = (
            outdir
            / f"training_curve_{args.baseset}_{args.training_mode}_{args.training_xaxis}.pdf"
        )

        # Assuming we update make_training_plot to handle dfs_by_method
        make_training_plot(
            dfs_by_method,
            outpath,
            mode=args.training_mode,
            xaxis=args.training_xaxis,
            window=args.training_window,
            include_deglex_reference=True,
            title=None,
            band="iqr",
        )
        print(f"Wrote training plot: {outpath}")

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = (root_dir / outdir).resolve()

    if td3_dfs:
        make_reward_ecdf_figs(
            td3_dfs,
            outdir,
            baseset=args.baseset,
            mode="pct",
        )
        print(f"Wrote reward ECDFs to {outdir}")

        if args.make_test_delta_plots:
            outdir = Path(args.outdir)
            if not outdir.is_absolute():
                outdir = (root_dir / outdir).resolve()

            make_test_delta_figs(
                td3_dfs,
                outdir,
                baseset=args.baseset,
                round_to=args.delta_round,
            )
            print(f"Wrote delta plots to {outdir}")

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = (root_dir / outdir).resolve()

    return dfs_by_method


if __name__ == "__main__":
    main()
