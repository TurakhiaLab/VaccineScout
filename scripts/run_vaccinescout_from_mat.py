#!/usr/bin/env python3
"""
Run VaccineScout for a single cutoff date straight from a MAT.

Pipeline
--------
1. Date every leaf, from ``--metadata``, ``--dates``, or the leaf name.
2. Keep leaves inside the circulating window (``analysis:`` in config.yaml), and
   read each one's lineage off the tree's own clade annotations.
3. Spike haplotype -> amino-acid sequence for those leaves only.
4. Collapse to distinct sequences; drop any seen on < ``analysis.min_tip_count``
   tips; weight the rest by tip count and carry their lineages along.
5. Score the cohort with CoVFit (reuses an existing predictions TSV).
6. Pairwise Spike distances via the ``pairwise_dist`` binary.
7. Weighted PAM for K = 1..``--k-max`` at ``analysis.beta``.
8. Write the CSV, each medoid labelled by the lineage of the tips behind it.

Usage
-----
    python scripts/run_vaccinescout_from_mat.py \
        --cutoff-date 2022-06-01 \
        --mat public-latest.all.masked.pb.gz \
        --reference NC_045512v2.fa \
        --output_dir results \
        --output_file vaccinescout_medoids.csv

    # dates from a metadata table instead of the leaf names (plain or gzipped)
    python scripts/run_vaccinescout_from_mat.py \
        --cutoff-date 2022-06-01 \
        --mat public-latest.all.masked.pb.gz \
        --reference NC_045512v2.fa \
        --metadata public-latest.metadata.tsv.gz \
        --output_dir results --output_file vaccinescout_medoids.csv

Leaf dating
-----------
``--metadata`` and ``--dates`` are both optional; without either, the date is
read from the leaf name (its trailing date field). Either file may be plain or gzipped.

In every mode a full ``YYYY-MM-DD`` is used as given and a ``YYYY-MM`` is taken
as the first of that month; a bare year or a placeholder is dropped. See
``_parse_dates``.

``--dates`` reads a chronumental ``predicted_date`` table. 
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import bte
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import get_amino_acid_mutations_of_sequences as gaam  # noqa: E402
from config_loader import load_config  # noqa: E402
from Helpers import apply_mutations_flat_list, compute_mutations  # noqa: E402
from lineage_spike_from_mat import (  # noqa: E402
    ref_spike_amino_acids,
    sort_spike_mutations,
)

# The medoid search, its fitness weighting and the distance-to-current metric
# all live in the step-10 report script. Importing them keeps one
# implementation of each. The module reads config.yaml at import time and pulls
# in openpyxl; matplotlib is imported lazily inside its plotting functions, so
# this stays cheap.
from vaccinescout_results_across_years import (  # noqa: E402
    evaluate_on_current_vectorized,
    load_covfit_scores,
    preprocess_for_weighted_pam_fast,
    run_covfit,
    weighted_pam,
)

CFG = load_config()
FITNESS_COLUMN = str(CFG.fitness_column)

# Third-party checkouts sit where the README's installation section puts them:
# beside this repository, in the working directory. They are deliberately not
# read from `paths:`, so a run needs nothing configured beyond the analysis
# parameters. (`pyro-cov` is found the same way by the module that imports
# `pyrocov`, which already searches the working directory and the repo root.)
REPO_ROOT = SCRIPT_DIR.parent
COVFIT_DIRNAME = "CoVFit_CLI"

#: Built into scripts/ by the CMake step, next to this file.
PAIRWISE_DIST_BIN = SCRIPT_DIR / "pairwise_dist"

#: Columns of the output CSV, in order.
OUTPUT_COLUMNS = [
    "k",
    "medoid",
    "covfit_fitness",
    "distance_current",
    # Last, because a merged label runs to hundreds of characters.
    "lineage",
]


def log(message: str) -> None:
    print(message, flush=True)


class Timer:
    """Print how long a step took, so a long run reports its own progress."""

    def __init__(self, label: str):
        self.label = label

    def __enter__(self) -> "Timer":
        self.start = time.time()
        log(f"\n=== {self.label} ===")
        return self

    def __exit__(self, *exc) -> None:
        log(f"  ({self.label}: {time.time() - self.start:.1f}s)")


# ---------------------------------------------------------------------------
# Step 1-2: date the leaves and keep the circulating window
# ---------------------------------------------------------------------------

def circulating_window(cutoff_date: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    The circulating window ``[lo, hi)`` for a cutoff.
    Mirrors the ``current_df`` half of ``Helpers.split_circulating_and_future``.
    """
    kwargs = CFG.split_kwargs()
    cutoff = pd.Timestamp(cutoff_date)

    hi = cutoff - pd.Timedelta(days=kwargs["left_buffer_days"])
    lo = hi
    if kwargs["circulation_months"] is not None:
        lo -= pd.DateOffset(months=kwargs["circulation_months"])
    if kwargs["circulation_days"] is not None:
        lo -= pd.Timedelta(days=kwargs["circulation_days"])
    return lo, hi


def _parse_dates(values: pd.Series) -> pd.Series:
    """
    Parse a date down to month precision, as NaT where that is not possible.

    Accepted, most precise first::

        2022-05-01                  -> 2022-05-01
        2019-12-07 20:39:59.139404  -> 2019-12-07   (chronumental timestamp)
        2022-06                     -> 2022-06-01   (first of that month)
        2022-06-99                  -> 2022-06-01   (day unreadable)

    A trailing time is allowed, because chronumental writes ``predicted_date``
    as ``2019-12-07 20:39:59.139404`` and
    ``create_from_chronumental_data.ipynb`` likewise keeps just the date.

    A day that cannot be read falls back to the month rather than being thrown
    away: ``2022-06-99``, ``2022-06-XX``, ``2022-06-1`` and ``2022-06-01extra``
    all give 2022-06-01, since the sample is still known to be from June. Only
    a value with no usable year-month survives as NaT -- a bare year, a
    placeholder, or an impossible month such as ``2022-13``.
    """
    text = values.str.strip()

    full = text.str.extract(r"^(\d{4}-\d{1,2}-\d{1,2})(?:[ T].*)?$", expand=False)
    exact = pd.to_datetime(full, format="%Y-%m-%d", errors="coerce")

    # The year-month prefix carries whatever the day could not: absent, out of
    # range, or not a number at all.
    month_prefix = text.str.extract(r"^(\d{4}-\d{1,2})", expand=False)
    month = pd.to_datetime(month_prefix, format="%Y-%m", errors="coerce")

    return exact.fillna(month)


def dates_from_leaf_names(leaf_ids: Sequence[str]) -> pd.Series:
    """
    Date every leaf from the trailing ``|YYYY-MM-DD`` field of its name.
    This is the dating ``create_from_metadata.ipynb`` uses. 
    """
    names = pd.Series(leaf_ids, dtype="object")
    tails = names.str.rsplit("|", n=1).str[-1]
    return pd.Series(_parse_dates(tails).values, index=names.values)


def _windowed_sequences_from_table(
    path: Path,
    strain_column: str,
    date_column: str,
    lo: pd.Timestamp,
    hi: pd.Timestamp,
    chunk_size: int = 2_000_000,
) -> pd.Series:
    """
    Stream a strain/date TSV and keep only the rows dated inside ``[lo, hi)``.
    """
    kept: List[pd.Series] = []
    n_rows = 0
    reader = pd.read_csv(
        path,
        sep="\t",
        usecols=[strain_column, date_column],
        dtype=str,
        chunksize=chunk_size,
    )
    for chunk in reader:
        n_rows += len(chunk)
        dates = _parse_dates(chunk[date_column])
        inside = dates.notna() & (dates >= lo) & (dates < hi)
        if inside.any():
            kept.append(
                pd.Series(
                    dates[inside].values,
                    index=chunk.loc[inside, strain_column].values,
                )
            )

    log(f"  read {n_rows:,} rows from {path.name}")
    if not kept:
        return pd.Series(dtype="datetime64[ns]")
    return pd.concat(kept)


def build_lineage_table(
    tree: bte.MATree,
    leaf_ids: Sequence[str],
    annotation_index: int = 1,
) -> pd.Series:
    """
    The lineage of every leaf in ``leaf_ids``, as a Series indexed by leaf id.
    NaN for a leaf with no lineage anywhere in its ancestry.
    """
    lineages: List[Optional[str]] = []
    report_every = max(1, len(leaf_ids) // 5)

    for i, leaf_id in enumerate(leaf_ids):
        labels = tree.get_node(leaf_id).most_recent_annotation()
        label = labels[annotation_index].strip() if len(labels) > annotation_index else ""
        lineages.append(label or None)

        if (i + 1) % report_every == 0:
            log(f"  {i + 1:,}/{len(leaf_ids):,} haplotypes annotated")

    table = pd.Series(lineages, index=pd.Index(leaf_ids), dtype="object")
    log(f"  {table.notna().sum():,} haplotype/lineage pair(s), "
        f"{table.nunique():,} distinct lineage(s)")
    return table


def resolve_windowed_leaves(
    tree: bte.MATree,
    cutoff_date: str,
    metadata: Optional[Path],
    dates_tsv: Optional[Path],
    annotation_index: int = 1,
) -> pd.DataFrame:
    """
    Every leaf on the tree dated inside the cutoff's circulating window, with
    its lineage.

    Returns a DataFrame indexed by leaf id with ``Date`` and ``Lineage``
    columns. Dates come from ``dates_tsv`` (chronumental) if given, else
    ``metadata``, else the leaf names.
    """
    lo, hi = circulating_window(cutoff_date)
    log(f"  Circulating Window: [{lo.date()}, {hi.date()})")

    if dates_tsv is not None:
        log(f"  Extracting dates from chronumental table {dates_tsv}")
        dated = _windowed_sequences_from_table(dates_tsv, "strain", "predicted_date", lo, hi)
    elif metadata is not None:
        log(f"  Extracting dates from metadata {metadata}")
        dated = _windowed_sequences_from_table(metadata, "strain", "date", lo, hi)
    else:
        log("  Extracting dates from haplotype names")
        dated = dates_from_leaf_names(tree.get_leaves_ids())
        dated = dated[dated.notna() & (dated >= lo) & (dated < hi)]

    # A metadata or chronumental row can name a strain that is not a leaf on
    # this tree (and chronumental also dates internal nodes), so intersect.
    log(f"  {len(dated):,} dated haplotypes inside the window")
    leaf_ids: Set[str] = set(tree.get_leaves_ids())
    log(f"  {len(leaf_ids):,} haplotypes on the tree")

    dated = dated[~dated.index.duplicated(keep="first")]
    windowed = dated[dated.index.isin(leaf_ids)]
    log(f"  {len(windowed):,} windowed haplotypes present on the tree")

    if windowed.empty:
        raise SystemExit(
            f"No haplotype on the tree is dated inside [{lo.date()}, {hi.date()}).\n"
            "Check --cutoff-date against the tree's date range, and that the "
            "strain names in any --metadata / --dates table match the leaf names."
        )

    frame = windowed.rename("Date").to_frame()
    frame["Lineage"] = build_lineage_table(tree, frame.index.tolist(), annotation_index)
    return frame


# ---------------------------------------------------------------------------
# Step 3: Spike sequences for the windowed leaves
# ---------------------------------------------------------------------------

def sequence_id(sequence: str) -> str:
    """
    A ``Sequence_ID`` derived from the sequence itself.
    """
    return "S_" + hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:12]


def spike_sequences_for_leaves(
    tree: bte.MATree,
    leaf_ids: Sequence[str],
    refseq: str,
    ref_spike_aa: str,
) -> List[str]:
    """
    The Spike amino-acid sequence of every leaf in ``leaf_ids``.
    """
    start, end = gaam.GENE_TO_POSITION["S"]
    cache: Dict[Tuple[str, ...], str] = {}
    sequences: List[str] = []

    report_every = max(1, len(leaf_ids) // 20)
    for i, leaf_id in enumerate(leaf_ids):
        spike_nt = [m for m in tree.get_haplotype(leaf_id) if start <= int(m[1:-1]) <= end]
        key = tuple(sorted(spike_nt))

        sequence = cache.get(key)
        if sequence is None:
            aa = gaam.nuc_mutations_to_aa_mutations_modified(refseq, list(key))
            sequence = apply_mutations_flat_list(sort_spike_mutations(aa), ref_spike_aa)
            cache[key] = sequence
        sequences.append(sequence)

        if (i + 1) % report_every == 0:
            log(f"  {i + 1:,}/{len(leaf_ids):,} haplotypes  ({len(cache):,} distinct haplotypes)")

    log(f"  translated {len(cache):,} distinct Spike haplotype(s)")
    return sequences


# ---------------------------------------------------------------------------
# Step 4: the circulating cohort
# ---------------------------------------------------------------------------

def build_cohort(
    sequences: Sequence[str],
    lineages: Sequence[Optional[str]],
    min_tip_count: int,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Collapse leaf sequences into the weighted circulating variants.

    ``sequences`` and ``lineages`` hold one entry per windowed leaf, and the
    ``value_counts`` below is what turns those repeats into a frequency:
    identical Spike sequences hash to the same ``Sequence_ID``, so the number
    of times an id appears is the number of tips carrying that variant.
    ``Weight`` normalises it over the cohort.

    Distinct lineages routinely share a Spike sequence -- a sublineage defined
    by a mutation outside Spike carries the same protein as its parent -- so
    the merged tips bring several lineage names with them. ``Lineage`` joins
    every name seen on the variant's tips with ``|``, most tips first, so the
    leading name is the variant's dominant lineage; names tying on tip count
    fall back to alphabetical order so a rerun labels the variant the same way.
    """
    ids = [sequence_id(s) for s in sequences]
    id_to_seq = dict(zip(ids, sequences))
    tips = pd.DataFrame({"Sequence_ID": ids, "Lineage": list(lineages)})

    cohort = (
        tips["Sequence_ID"]
        .value_counts()
        .rename_axis("Sequence_ID")
        .reset_index(name="Tip_Count")
        .sort_values("Tip_Count", ascending=False)
        .reset_index(drop=True)
    )
    log(f"  {len(cohort):,} distinct Spike sequence(s) over {len(sequences):,} tips")

    cohort = cohort[cohort["Tip_Count"] >= min_tip_count].reset_index(drop=True)
    log(f"  {len(cohort):,} kept at Tip_Count >= {min_tip_count}")
    if cohort.empty:
        raise SystemExit(
            f"No sequence reaches min_tip_count={min_tip_count} in this window. "
            "Widen analysis.circulation_days or lower --min-tip-count."
        )

    # Weight over the surviving cohort.
    cohort["Weight"] = cohort["Tip_Count"] / cohort["Tip_Count"].sum()

    # Every distinct lineage seen on a surviving variant's tips, merged into one
    # name. Counting the tips behind each (variant, lineage) pair collapses the
    # per-tip rows the same way a de-duplication would, but keeps the count to
    # sort on, so the merged label reads as a ranking rather than an alphabet.
    kept = tips[tips["Sequence_ID"].isin(set(cohort["Sequence_ID"]))]
    pairs = (
        kept.loc[kept["Lineage"].notna(), ["Sequence_ID", "Lineage"]]
        .groupby(["Sequence_ID", "Lineage"], sort=False)
        .size()
        .reset_index(name="Pair_Tip_Count")
        .sort_values(
            ["Sequence_ID", "Pair_Tip_Count", "Lineage"],
            ascending=[True, False, True],
        )
    )
    merged = pairs.groupby("Sequence_ID", sort=False)["Lineage"].agg("|".join)
    cohort["Lineage"] = cohort["Sequence_ID"].map(merged)

    unlabelled = int(cohort["Lineage"].isna().sum())
    if unlabelled:
        log(f"  {unlabelled:,} variant(s) have no lineage on any of their tips")
    log(f"  {len(pairs):,} (variant, lineage) pair(s) over {len(cohort):,} variants")

    id_to_seq = {sid: id_to_seq[sid] for sid in cohort["Sequence_ID"]}
    return cohort, id_to_seq


def write_cohort_lineages(cohort: pd.DataFrame, path: Path) -> Path:
    """Persist each circulating variant's merged lineage name beside the CSV."""
    columns = ["Sequence_ID", "Tip_Count", "Weight", "Lineage"]
    cohort[columns].to_csv(path, sep="\t", index=False)
    log(f"  wrote {len(cohort):,} variant/lineage row(s) to {path.name}")
    return path


# ---------------------------------------------------------------------------
# Steps 5-6: CoVFit fitness and pairwise distances
# ---------------------------------------------------------------------------

def write_fasta(cohort: pd.DataFrame, id_to_seq: Dict[str, str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for seq_id in cohort["Sequence_ID"]:
            handle.write(f">{seq_id}\n{id_to_seq[seq_id]}\n")
    log(f"  wrote {len(cohort):,} sequence(s) to {path}")
    return path


def write_medoid_fastas(
    medoids_by_k: Dict[int, List[str]],
    id_to_seq: Dict[str, str],
    directory: Path,
) -> List[Path]:
    """
    One FASTA per K, holding that K's medoid Spike sequences.

    The header is the medoid's ``Sequence_ID`` and nothing else. That is the
    same string the report CSV's ``medoid`` column carries, and the same key
    used by ``current_id_to_seq.parquet`` and ``current_seq_df.fasta``, so a
    record here joins onto any of them without parsing.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for k, medoid_ids in sorted(medoids_by_k.items()):
        path = directory / f"medoids_k{k}.fasta"
        with open(path, "w", encoding="utf-8") as handle:
            for medoid in medoid_ids:
                handle.write(f">{medoid}\n{id_to_seq[medoid]}\n")
        written.append(path)
    log(f"  wrote {len(written)} medoid FASTA(s) to {directory}")
    return written


def write_medoid_mutations(
    medoids_by_k: Dict[int, List[str]],
    id_to_seq: Dict[str, str],
    ref_spike_aa: str,
    path: Path,
) -> Path:
    """
    Every medoid selected at any valency, once, with its Spike mutations.

    A medoid picked at K=1 is usually picked again at every larger K, so the
    per-K FASTAs and the report both repeat it. Here each appears once, in the
    order it first enters a medoid set, which is roughly decreasing importance.

    Mutations are against the reference Spike, in site order, formatted
    ``S:<ref><site><new>`` by ``Helpers.compute_mutations`` -- joined with 
    ``|`` as the report's ``lineage`` column is, so no field needs quoting.
    """
    seen: List[str] = []
    for k in sorted(medoids_by_k):
        for medoid in medoids_by_k[k]:
            if medoid not in seen:
                seen.append(medoid)

    frame = pd.DataFrame(
        [
            {
                "medoid": medoid,
                "mutations": "|".join(compute_mutations(id_to_seq[medoid], ref_spike_aa)),
            }
            for medoid in seen
        ],
        columns=["medoid", "mutations"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    log(f"  wrote {len(frame)} unique medoid(s) to {path}")
    return path


def covfit_dir(required: bool = True) -> Optional[Path]:
    """
    The ``CoVFit_CLI`` directory, taken from the working directory or the repo
    root rather than from ``paths.covfit_cli``.

    ``required=False`` returns None rather than exiting, so a run can carry on
    without fitness estimates instead of stopping on a missing optional tool.
    """
    executable = Path(str(CFG.get("covfit", "executable", "./covfit_cli"))).name
    candidates = (Path.cwd() / COVFIT_DIRNAME, REPO_ROOT / COVFIT_DIRNAME)
    for candidate in candidates:
        if (candidate / executable).is_file():
            return candidate
    if not required:
        return None
    raise SystemExit(
        f"Could not find {executable} in a {COVFIT_DIRNAME}/ directory. Looked in:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nExtract the CoVFit CLI archive here, as the README's installation "
          "section describes."
    )


def covfit_scores(
    cohort: pd.DataFrame,
    fasta_path: Path,
    work_dir: Path,
) -> pd.DataFrame:
    """
    CoVFit scores for the cohort.

    ``run_covfit`` returns an existing ``CoVFit_Predictions_fold_<n>.tsv``
    untouched, so a repeat run against the same work dir costs nothing.
    """
    tsv_path = run_covfit(fasta_path, work_dir / "current_covfit", covfit_dir())

    scores = load_covfit_scores(tsv_path)
    if FITNESS_COLUMN not in scores.columns:
        raise SystemExit(
            f"Column {FITNESS_COLUMN!r} not found in {tsv_path}; "
            f"available: {list(scores.columns)}"
        )

    missing = set(cohort["Sequence_ID"]) - set(scores["Sequence_ID"])
    if missing:
        raise SystemExit(
            f"{len(missing)} cohort sequence(s) have no CoVFit score in {tsv_path}, "
            f"e.g. {sorted(missing)[:3]}.\n"
            "A predictions file from another run keys on that run's Sequence_IDs."
        )
    log(f"  {len(scores):,} CoVFit score(s) available")
    return scores


def pairwise_distances(
    cohort: pd.DataFrame,
    id_to_seq: Dict[str, str],
    work_dir: Path,
) -> pd.DataFrame:
    """
    All-pairs Spike substitution distances from the ``pairwise_dist`` binary.
    The binary emits N*N rows and writes inside ``work_dir``.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    seq_parquet = work_dir / "current_id_to_seq.parquet"
    dist_parquet = work_dir / "pairwise_dist.parquet"

    seq_ids = cohort["Sequence_ID"].tolist()
    pq.write_table(
        pa.table(
            {
                "key": pa.array(seq_ids, type=pa.string()),
                "value": pa.array([id_to_seq[s] for s in seq_ids], type=pa.string()),
            }
        ),
        str(seq_parquet),
    )

    if not PAIRWISE_DIST_BIN.is_file():
        raise SystemExit(
            f"pairwise_dist not found at {PAIRWISE_DIST_BIN}. Build it from the "
            "repo root:\n"
            "    cmake -S scripts -B scripts/build -DCMAKE_BUILD_TYPE=Release\n"
            "    cmake --build scripts/build -j"
        )
    binary = PAIRWISE_DIST_BIN
    cmd = [str(binary.resolve()), "--parquet", str(seq_parquet), "--output", str(dist_parquet)]
    log(f"  running {binary.name} on {len(seq_ids):,} sequences")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(result.stdout[-2000:])
        log(result.stderr[-2000:])
        raise RuntimeError(f"pairwise_dist failed with exit code {result.returncode}")

    dist_df = pd.read_parquet(dist_parquet, engine="pyarrow")
    log(f"  {len(dist_df):,} pair(s) -> {dist_parquet.name}")
    return dist_df


# ---------------------------------------------------------------------------
# Steps 7-8: medoid selection and the report
# ---------------------------------------------------------------------------

def select_medoids(
    cohort: pd.DataFrame,
    dist_df: pd.DataFrame,
    scores: Optional[pd.DataFrame],
    beta: Optional[float],
    k_max: int,
) -> Dict[int, List[str]]:
    """
    Weighted PAM medoids for K = 1..k_max, weighted by tips * e**(beta*fitness).

    ``scores=None`` drops the fitness term and leaves tip prevalence as the only
    weight, which is what ``effective_weights`` already does when either the
    scores or beta are absent.
    """
    dist_matrix, weights, _id2idx, idx2id, fitness_scores = preprocess_for_weighted_pam_fast(
        cohort,
        dist_df,
        use_covfit_scores=scores is not None,
        covfit_score_df=scores,
    )

    k_max = min(k_max, len(cohort))
    medoids_by_k: Dict[int, List[str]] = {}
    for k in range(1, k_max + 1):
        medoids, _labels = weighted_pam(
            dist_matrix,
            weights,
            k=k,
            fitness_scores=fitness_scores,
            beta=beta,
        )
        medoids_by_k[k] = [idx2id[i] for i in medoids]
        log(f"  K={k}: {', '.join(medoids_by_k[k])}")
    return medoids_by_k


def build_report(
    medoids_by_k: Dict[int, List[str]],
    cohort: pd.DataFrame,
    id_to_seq: Dict[str, str],
    beta: float,
) -> pd.DataFrame:
    """
    One row per (K, medoid), carrying its lineage, CoVFit score and the
    fitness-weighted distance from the circulating cohort to that K's medoid set.

    A medoid is one of the circulating variants, so its lineage is already on
    its cohort row -- read off the tips it was built from, not matched back to
    a lineage's Spike sequence afterwards.
    """
    info = cohort.set_index("Sequence_ID")

    rows: List[dict] = []
    for k, medoid_ids in sorted(medoids_by_k.items()):
        distance_current = evaluate_on_current_vectorized(
            candidate_seqs=[id_to_seq[m] for m in medoid_ids],
            current_seq_df=cohort,
            id_to_seq=id_to_seq,
            beta=beta,
        )

        for medoid in medoid_ids:
            variant = info.loc[medoid]
            rows.append(
                {
                    "k": k,
                    "medoid": medoid,
                    "covfit_fitness": variant.get(FITNESS_COLUMN, np.nan),
                    "distance_current": distance_current,
                    "lineage": variant["Lineage"],
                }
            )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def write_report(report: pd.DataFrame, path: Path) -> Path:
    """
    Write the medoid CSV with a rule between one K's medoid set and the next.

    The merged lineage labels run to hundreds of characters, which makes the
    boundary between one K's set and the next hard to find by eye.

    The rule is a one-field row, so the file is no longer a uniform CSV: pandas
    reads it, but ``k`` comes back as object dtype with ``'----'`` in it, which
    makes ``notna()`` useless for dropping the rules. Read it with::

        df = pd.read_csv(path)
        df = df[pd.to_numeric(df["k"], errors="coerce").notna()]
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rule = "-" * len(OUTPUT_COLUMNS)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(OUTPUT_COLUMNS) + "\n")
        for _k, block in report.groupby("k", sort=True):
            block.to_csv(handle, header=False, index=False, lineterminator="\n")
            handle.write(rule + "\n")

    log(f"  wrote {len(report)} row(s) to {path}")
    return path


def summarise_lineage(label, limit: int = 5) -> str:
    """
    The ``limit`` most abundant names of a merged lineage label, for the closing
    summary. ``build_cohort`` writes the label most tips first, so truncating it
    keeps the top lineages that carry the variant.
    """
    if not isinstance(label, str) or not label:
        return "(no lineage on its tips)"
    names = label.split("|")
    if len(names) <= limit:
        return "|".join(names)
    return "|".join(names[:limit]) + f"  (+{len(names) - limit} more)"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select VaccineScout medoids for the cutoff date by analyzing sequences from the MAT."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mat",
        required=True,
        help="Mutation Annotated Tree (.pb / .pb.gz).",
    )
    parser.add_argument(
        "--cutoff-date",
        required=True,
        help="Cutoff date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help=(
            "Metadata TSV with 'strain' and 'date' columns."
        ),
    )
    parser.add_argument(
        "--dates",
        default=None,
        help=(
            "chronumental table with 'strain' and 'predicted_date'. Takes "
            "precedence over --metadata."
        ),
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="SARS-CoV-2 reference FASTA, 29903 bp (NC_045512v2 / Wuhan-Hu-1).",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        required=True,
        dest="output_dir",
        help=(
            "Directory for the output medoid CSV and therun's intermediates."
        ),
    )
    parser.add_argument(
        "--output_file",
        "--output-file",
        required=True,
        dest="output_file",
        help="Name of the medoid CSV, written inside --output_dir.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Fitness weighting exponent. Default: analysis.beta.",
    )
    parser.add_argument(
        "--min-tip-count",
        type=int,
        default=None,
        help="Drop variants with fewer sequences. Default: analysis.min_tip_count.",
    )
    parser.add_argument("--k-max", type=int, default=8, help="Select medoids for K = 1..k-max.")
    args = parser.parse_args()

    # Every path is explicit.
    args.beta = CFG.beta if args.beta is None else args.beta
    args.min_tip_count = CFG.min_tip_count if args.min_tip_count is None else args.min_tip_count
    args.annotation_index = int(CFG.get("mat_scripts", "annotation_index", 1))

    args.output = str(Path(args.output_dir) / Path(args.output_file).name)

    for attr in ("mat", "reference", "output_dir", "output",
                 "metadata", "dates"):
        value = getattr(args, attr, None)
        if value:
            setattr(args, attr, str(Path(value).expanduser().resolve()))
    return args


def main() -> None:
    args = parse_args()
    started = time.time()

    for label, path in (("MAT", args.mat), ("reference", args.reference)):
        if not Path(path).is_file():
            raise SystemExit(f"{label} not found: {path}")

    work_dir = Path(args.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    log(f"cutoff date : {args.cutoff_date}")
    log(f"beta        : {args.beta}")
    log(f"output dir  : {work_dir}")

    with Timer("Loading Reference sequence and MAT"):
        refseq = gaam.load_reference_sequence_modified(args.reference)
        ref_spike_aa = ref_spike_amino_acids(refseq)
        log(f"  Reference Spike: {len(ref_spike_aa)} aa")
        tree = bte.MATree(args.mat)

    with Timer("Dating and annotating haplotypes, selecting the circulating window"):
        windowed = resolve_windowed_leaves(
            tree,
            args.cutoff_date,
            Path(args.metadata) if args.metadata else None,
            Path(args.dates) if args.dates else None,
            args.annotation_index,
        )

    with Timer("Extracting Spike sequence of haplotypes"):
        sequences = spike_sequences_for_leaves(
            tree, windowed.index.tolist(), refseq, ref_spike_aa
        )

    with Timer("Building circulating variants data"):
        cohort, id_to_seq = build_cohort(
            sequences, windowed["Lineage"].tolist(), args.min_tip_count
        )
        write_cohort_lineages(cohort, work_dir / "cohort_lineages.tsv")

    with Timer("Writing the FASTA for circulating variants"):
        fasta_path = write_fasta(cohort, id_to_seq, work_dir / "current_seq_df.fasta")

    # CoVFit is optional. Without it there is no fitness term to weight by, and
    # beta goes with it: effective_weights() falls back to tip prevalence when
    # either is None, so selection continues on prevalence alone rather than
    # stopping. FITNESS_COLUMN is still filled, with NaN, because build_report
    # reports it and evaluate_on_current_vectorized reads it before handing it to
    # effective_weights.
    covfit_root = covfit_dir(required=False)
    if covfit_root is None:
        scores = None
        beta = None
        cohort[FITNESS_COLUMN] = np.nan
        log("")
        log("*** CoVFit is NOT available. VaccineScout will NOT consider "
            "fitness estimates. ***")
        log("*** Vaccine Candidates are selected based on current prevalence alone, and the"
            "covfit_fitnesscolumn will be empty. ***")
        log("")
    else:
        beta = args.beta
        with Timer("Scoring the variants with CoVFit"):
            scores = covfit_scores(cohort, fasta_path, work_dir)
            score_map = scores.set_index("Sequence_ID")[FITNESS_COLUMN]
            cohort[FITNESS_COLUMN] = cohort["Sequence_ID"].map(score_map)

    with Timer("Computing pairwise Spike distances"):
        dist_df = pairwise_distances(cohort, id_to_seq, work_dir)

    beta_label = "none (no fitness)" if beta is None else f"{beta}"
    with Timer(f"Selecting medoids (K = 1..{args.k_max}, beta = {beta_label})"):
        medoids_by_k = select_medoids(cohort, dist_df, scores, beta, args.k_max)

    with Timer("Writing the report"):
        report = build_report(medoids_by_k, cohort, id_to_seq, beta)
        write_report(report, Path(args.output))
        medoid_dir = work_dir / "medoids"
        write_medoid_fastas(medoids_by_k, id_to_seq, medoid_dir)
        write_medoid_mutations(
            medoids_by_k, id_to_seq, ref_spike_aa,
            medoid_dir / "medoids_mutations.csv",
        )

    log(f"\nDone in {(time.time() - started) / 60:.1f} min.")
    log(f"\nMedoid lineages by K (up to 5 names each; full list in "
        f"{Path(args.output).name}):")
    for k in sorted(medoids_by_k):
        rows = report[report["k"] == k]
        log(f"  K={k}  distance_to_current={rows['distance_current'].iloc[0]:.4f}")
        for medoid, lineage in zip(rows["medoid"], rows["lineage"]):
            log(f"      {medoid}  {summarise_lineage(lineage)}")


if __name__ == "__main__":
    main()
