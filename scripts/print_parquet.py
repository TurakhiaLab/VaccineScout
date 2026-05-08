#!/usr/bin/env python3
"""Read a Parquet file and print a short preview (first 50 rows by default; safe for large files)."""
import argparse
import re
import sys
from typing import Optional

import numpy as np
import pandas as pd


#DEFAULT_PATH = "/data/pgangwar/vaccinescout/data/all_nodes_spike_2025-09-01.parquet"
#DEFAULT_PATH =  "/data/sukhanna/MPP/all_node_spike_mutations_2025_09_01.parquet"

DEFAULT_PATH = "/data/pgangwar/vaccinescout/data/all_leaves_spike_2025-09-01.parquet"

_MUTATION_COLUMN_CANDIDATES = ("aa_mutations", "node_spike_mutations")


def _resolve_mutations_column(
    df: pd.DataFrame, mutations_col: Optional[str]
) -> str:
    """
    Pick the mutations column: explicit name if given, else first of
    ``aa_mutations`` / ``node_spike_mutations`` present in ``df``.
    """
    if mutations_col is not None:
        if mutations_col not in df.columns:
            raise KeyError(
                f"Mutations column {mutations_col!r} not found; "
                f"available: {list(df.columns)}"
            )
        return mutations_col
    for name in _MUTATION_COLUMN_CANDIDATES:
        if name in df.columns:
            return name
    raise KeyError(
        f"No mutations column found (tried {list(_MUTATION_COLUMN_CANDIDATES)}); "
        f"available: {list(df.columns)}"
    )


_RE_EMPTY_BRACKETS = re.compile(r"^\s*[\[(]\s*[\])]\s*$")


def _mutation_cell_nonempty(val) -> bool:
    """True if the cell encodes at least one mutation (non-empty string, list, etc.)."""
    if val is None:
        return False
    if isinstance(val, (list, tuple, set, frozenset)):
        return len(val) > 0
    if isinstance(val, np.ndarray):
        return val.size > 0
    try:
        if pd.isna(val):
            return False
    except (ValueError, TypeError):
        pass
    if isinstance(val, (bytes, bytearray)):
        return len(val) > 0
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null", "<na>"):
        return False
    if s in ("[]", "()", "{}"):
        return False
    if _RE_EMPTY_BRACKETS.match(s):
        return False
    return True


def export_top_leaf_haplotypes_csv(
    df: pd.DataFrame,
    out_csv: str,
    *,
    sample_id_col: str = "sample_id",
    mutations_col: Optional[str] = None,
    top_n: int = 10000,
) -> pd.DataFrame:
    """
    Keep only leaf haplotypes (exclude internal nodes whose IDs start with ``node_``),
    drop rows with no Spike mutations (empty string, empty list, or string ``[]``),
    sort by ``sample_id`` (lexicographic on stripped strings), and write the first
    ``top_n`` rows to ``out_csv``.

    ``mutations_col`` defaults to auto-detection: ``aa_mutations`` or
    ``node_spike_mutations``, whichever exists.

    Returns the DataFrame that was written (at most ``top_n`` rows).
    """
    if sample_id_col not in df.columns:
        raise KeyError(
            f"Missing column {sample_id_col!r}; available: {list(df.columns)}"
        )
    mut_col = _resolve_mutations_column(df, mutations_col)

    sid = df[sample_id_col].astype(str)
    mut_nonempty = df[mut_col].map(_mutation_cell_nonempty)
    filtered = df[~sid.str.startswith("node_") & mut_nonempty].copy()
    # Lexicographic order on string sample_id (strip whitespace; stable sort).
    filtered = filtered.sort_values(
        sample_id_col,
        key=lambda s: s.astype(str).str.strip(),
        kind="mergesort",
        na_position="last",
    ).head(top_n)
    filtered.to_csv(out_csv, index=False)
    return filtered


def read_parquet(path: str) -> pd.DataFrame:
    """Load Parquet with PyArrow; fall back to fastparquet on histogram validation bugs."""
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except FileNotFoundError:
        raise
    except OSError as exc:
        err = str(exc).lower()
        if "level histogram" not in err:
            raise
        try:
            return pd.read_parquet(path, engine="fastparquet")
        except ImportError:
            raise OSError(
                f"{exc}\n\nPyArrow 19.0.0 mis-reads some Parquet files; "
                "upgrade with `pip install 'pyarrow>=19.0.1'` or install "
                "`fastparquet` (`conda install -c conda-forge fastparquet`) "
                "as a fallback engine."
            ) from exc


def main() -> None:
    p = argparse.ArgumentParser(description="Preview a Parquet file.")
    p.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_PATH,
        help=f"Parquet path (default: {DEFAULT_PATH})",
    )
    p.add_argument(
        "-n",
        "--head",
        type=int,
        default=50,
        help="Rows to show (default: 50). Use 0 for schema + shape only.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Print the full DataFrame (can be huge; not recommended).",
    )
    p.add_argument(
        "--top-csv",
        metavar="PATH",
        help=(
            "After loading, drop internal nodes (node_*), rows with no mutations, "
            "sort by sample_id, and write the first --top-n rows to this CSV."
        ),
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=10000,
        help="Row cap for --top-csv (default: 10000).",
    )
    p.add_argument(
        "--mutations-col",
        metavar="NAME",
        default=None,
        help=(
            "Column with mutation strings for --top-csv (default: auto: aa_mutations "
            "or node_spike_mutations)."
        ),
    )
    args = p.parse_args()

    written: Optional[pd.DataFrame] = None

    try:
        df = read_parquet(args.path)
    except FileNotFoundError:
        print(f"File not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    if args.top_csv:
        written = export_top_leaf_haplotypes_csv(
            df,
            args.top_csv,
            top_n=args.top_n,
            mutations_col=args.mutations_col,
        )
        print(
            f"Wrote {len(written)} row(s) to {args.top_csv} "
            f"(top {args.top_n} after filter/sort).",
            file=sys.stderr,
        )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 80)

    print(df.dtypes, end="\n\n")
    print(f"shape: {df.shape}", end="\n\n")

    if args.all:
        print(df.to_string())
    elif args.head > 0:
        preview_df = written if written is not None else df
        print(preview_df.head(args.head).to_string())


if __name__ == "__main__":
    main()
