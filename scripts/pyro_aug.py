#!/usr/bin/env python3
"""
Expand the released PyR0 mutation table into the per-substitution Spike scores
the later notebooks read.

The PyR0 release lists a growth-rate ratio (``R / R_A``) for the substitutions it
observed. The notebooks need a score for any substitution they might encounter at
a Spike site, so this script closes the table over each site:

- **inverses** — ``S:XnY`` implies ``S:YnX`` with score ``1 / score(XnY)``;
- **chains** — ``X->Y`` and ``Y->Z`` at the same site imply ``X->Z`` with score
  ``score(X->Y) * score(Y->Z)``, applied repeatedly until nothing new appears.

Measured values always win over inferred ones, and self-transitions (``X->X``)
are never emitted.

Usage:
    python scripts/pyro_aug.py

Every argument defaults to the matching entry in ``config.yaml`` (the
``fitness:`` block), so the command above is usually all you need. An explicit
flag always overrides the configured value.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import load_config_optional  # noqa: E402

# Config is optional: both paths can still be supplied by flag, so the script
# runs in an environment that lacks PyYAML or config.yaml.
CFG, CFG_ERROR = load_config_optional()

SCORE_COLUMN = "R / R_A"
MUTATION_COLUMN = "mutation"

#: Spike substitution as written by PyR0, e.g. ``S:G339D``.
MUTATION_PATTERN = re.compile(r"^S:([A-Za-z])(\d+)([A-Za-z])$")


def expand_mutations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Close the mutation table over each Spike site.

    Takes a frame with ``mutation`` and ``R / R_A`` columns and returns one with
    the same columns, containing every substitution reachable by inverting or
    chaining the measured ones. Rows that are not parseable Spike substitutions
    are dropped.
    """
    df = df.dropna(subset=[MUTATION_COLUMN, SCORE_COLUMN]).copy()
    df[SCORE_COLUMN] = pd.to_numeric(df[SCORE_COLUMN], errors="coerce")
    df = df.dropna(subset=[SCORE_COLUMN]).copy()

    parsed_rows = []
    for _, row in df.iterrows():
        match = MUTATION_PATTERN.match(row[MUTATION_COLUMN])
        if not match:
            continue
        parsed_rows.append((
            int(match.group(2)),        # site
            match.group(1).upper(),     # old aa
            match.group(3).upper(),     # new aa
            float(row[SCORE_COLUMN]),
        ))

    # Measured edges, kept separate so they always take precedence.
    orig_edges = {(site, old, new): score for site, old, new, score in parsed_rows}
    edges = dict(orig_edges)

    # Inverses.
    for (site, a, b), score in list(orig_edges.items()):
        if a == b or score == 0:
            continue
        inverse = (site, b, a)
        if inverse not in edges:
            edges[inverse] = 1.0 / score

    # Transitive closure within each site, repeated until it stops growing.
    made_progress = True
    while made_progress:
        made_progress = False

        edges_by_site = defaultdict(list)
        for (site, x, y), score in edges.items():
            edges_by_site[site].append((x, y, score))

        for site, edge_list in edges_by_site.items():
            outgoing = defaultdict(list)
            for x, y, score in edge_list:
                outgoing[x].append((y, score))

            for x, y, score_xy in edge_list:
                for z, score_yz in outgoing.get(y, []):
                    if z == x:
                        continue
                    key = (site, x, z)
                    if key in edges:
                        continue
                    score = score_xy * score_yz
                    if not np.isfinite(score):
                        continue
                    edges[key] = float(score)
                    made_progress = True

    rows = []
    for (site, a, b), score in edges.items():
        if a == b:
            continue
        rows.append({
            MUTATION_COLUMN: f"S:{a}{site}{b}",
            SCORE_COLUMN: orig_edges.get((site, a, b), score),
        })

    return (pd.DataFrame(rows)
            .drop_duplicates(subset=[MUTATION_COLUMN])
            .reset_index(drop=True))


def check(df: pd.DataFrame) -> int:
    """Warn about anything malformed in the expanded table; return a count."""
    problems = 0

    duplicates = df[df.duplicated(subset=[MUTATION_COLUMN], keep=False)]
    if not duplicates.empty:
        print(f"Warning: {len(duplicates)} duplicate mutation(s):", file=sys.stderr)
        print(duplicates, file=sys.stderr)
        problems += len(duplicates)

    self_transitions = df[df[MUTATION_COLUMN].str[2] == df[MUTATION_COLUMN].str[-1]]
    if not self_transitions.empty:
        print(f"Warning: {len(self_transitions)} self transition(s):", file=sys.stderr)
        print(self_transitions, file=sys.stderr)
        problems += len(self_transitions)

    strict = re.compile(r"^S:[A-Z]\d+[A-Z]$")
    invalid = df[~df[MUTATION_COLUMN].str.match(strict)]
    if not invalid.empty:
        print(f"Warning: {len(invalid)} invalid mutation string(s):", file=sys.stderr)
        print(invalid, file=sys.stderr)
        problems += len(invalid)

    if df[SCORE_COLUMN].isna().any():
        print("Warning: NaN scores detected", file=sys.stderr)
        problems += 1
    if not np.issubdtype(df[SCORE_COLUMN].dtype, np.number):
        print("Warning: scores are not numeric", file=sys.stderr)
        problems += 1

    return problems


def per_site_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per Spike site: how many substitutions, and their score range."""
    sites = df[MUTATION_COLUMN].str[3:-1].astype(int)
    return df.assign(site=sites).groupby("site").agg(
        num_mutations=(MUTATION_COLUMN, "count"),
        min_score=(SCORE_COLUMN, "min"),
        max_score=(SCORE_COLUMN, "max"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand the PyR0 Spike mutation table with inverse and chained "
            "substitutions, and pickle the result for the analysis notebooks."
        )
    )
    parser.add_argument(
        "--pyro-version",
        default=None,
        dest="pyro_version",
        help=(
            "PyR0 release directory under <data_root>/<fitness.pyro_subdir> "
            "(default: fitness.pyro_version). Sets both paths below."
        ),
    )
    parser.add_argument(
        "--mutations",
        default=None,
        metavar="PATH",
        help="PyR0 mutation table TSV (default: fitness.pyro_mutations_filename).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="PATH",
        help="Pickle to write (default: fitness.pyro_spike_filename).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Also print the per-site summary table (one row per Spike site).",
    )
    return _apply_config_defaults(parser.parse_args())


def _apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill unset arguments from config.yaml; an explicit flag always wins."""
    version: Optional[str] = args.pyro_version
    if version is None and CFG is not None:
        version = CFG.pyro_version

    defaults = {
        "mutations": ("fitness.pyro_mutations_filename",
                      lambda c: c.pyro_mutations_tsv(version)),
        "output": ("fitness.pyro_spike_filename",
                   lambda c: c.pyro_spike_pickle(version)),
    }

    missing = []
    for dest, (config_key, getter) in defaults.items():
        if getattr(args, dest, None):
            continue
        value = None
        if CFG is not None:
            try:
                value = getter(CFG)
            except Exception:  # noqa: BLE001 - fall through to the error below
                value = None
        if value:
            setattr(args, dest, str(value))
        else:
            flag = "--" + dest.replace("_", "-")
            missing.append(f"{flag} (or set {config_key} in config.yaml)")

    if missing:
        detail = "" if CFG is not None else f"\nConfig unavailable: {CFG_ERROR}"
        raise SystemExit("Missing required input(s):\n  " + "\n  ".join(missing) + detail)

    args.pyro_version = version
    return args


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.mutations):
        raise FileNotFoundError(f"PyR0 mutation table not found: {args.mutations}")

    print(f"Reading PyR0 mutation table: {args.mutations}")
    df = pd.read_csv(args.mutations, sep="\t")
    df = df[[MUTATION_COLUMN, SCORE_COLUMN]]
    df = df.dropna(subset=[MUTATION_COLUMN])
    df = df[df[MUTATION_COLUMN].str.startswith("S:")]
    print(f"{len(df)} measured Spike substitution(s)")

    expanded = expand_mutations(df)
    print(f"{len(expanded)} substitution(s) after adding inverses and chains")

    check(expanded)
    if args.summary:
        print(per_site_summary(expanded))

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    expanded.to_pickle(output)
    print(f"Saved {len(expanded)} expanded spike scores to {output}")


if __name__ == "__main__":
    main()
