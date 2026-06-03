#!/usr/bin/env python3
"""
Discover every lineage root on a MAT via depth-first expansion, translate cumulative
nucleotide haplotypes to Spike amino-acid mutations, merge lineages that share the
same Spike mutation set, and write them to a single ``lineage_spike.fasta`` file.

A lineage root is the first internal node encountered in DFS whose
``node.annotations[annotation_index]`` equals that lineage label (same convention as
``lineage_spike_from_mat.py``).

Each FASTA record header lists every lineage that shares that Spike mutation set,
joined with ``|`` (sorted). Spike sequences are built by applying the mutation set to
``--ref-spike`` (default ``data/ref_spike_genome.fa``).

Requires: ``bte``, local ``pyro-cov`` checkout, repo-root ``Helpers.py``.

Example:
    python scripts/get_lineage_fasta.py \\
        --mat /path/to/tree.pb.gz \\
        --reference /path/to/NC_045512v2.fa \\
        --output-dir data
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import bte

_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
for _p in (_script_dir, _repo_root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import get_amino_acid_mutations_of_sequences as gaam  # noqa: E402
from Helpers import apply_mutations_flat_list, getReference  # noqa: E402
from lineage_spike_from_mat import (  # noqa: E402
    lineage_from_node,
    sort_spike_mutations,
)


def spike_mutations_at_node(tree: bte.MATree, refseq: str, node_id: str) -> List[str]:
    """Cumulative path mutations on the node, translated to ordered Spike AA calls."""
    nt_path = list(tree.get_haplotype(node_id))
    aa_path = gaam.nuc_mutations_to_aa_mutations_modified(refseq, nt_path)
    spike_muts = [m for m in aa_path if m.startswith("S:")]
    return sort_spike_mutations(spike_muts)


def collect_lineage_root_rows(
    tree: bte.MATree,
    refseq: str,
    *,
    annotation_index: int = 1,
    wanted: Optional[Set[str]] = None,
) -> List[dict]:
    """
    DFS over internal nodes; keep the first (root) hit per lineage label.

    Returns one row per lineage root with keys:
        lineage, node_id, spike_mutations (list), spike_key (tuple)
    """
    seen_lineages: Set[str] = set()
    rows: List[dict] = []

    for node in tree.depth_first_expansion():
        if node.is_leaf():
            continue

        lineage = lineage_from_node(node, annotation_index)
        if lineage is None or lineage in seen_lineages:
            continue
        if wanted is not None and lineage not in wanted:
            continue

        spike_ordered = spike_mutations_at_node(tree, refseq, node.id)
        rows.append(
            {
                "lineage": lineage,
                "node_id": node.id,
                "spike_mutations": spike_ordered,
                "spike_key": tuple(spike_ordered),
            }
        )
        seen_lineages.add(lineage)

    return rows


def merge_lineages_by_spike_mutations(
    rows: Sequence[dict],
) -> Dict[Tuple[str, ...], List[str]]:
    """Group lineage names by identical ordered Spike mutation tuples."""
    groups: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    for row in rows:
        groups[row["spike_key"]].append(row["lineage"])
    for key in groups:
        groups[key] = sorted(set(groups[key]))
    return dict(groups)


def lineage_group_header(lineages: Sequence[str]) -> str:
    return "|".join(sorted(lineages))


LINEAGE_SPIKE_FASTA = "lineage_spike.fasta"


def write_lineage_spike_fasta(
    groups: Dict[Tuple[str, ...], List[str]],
    ref_spike_aa: str,
    output_dir: Path,
) -> int:
    """Write all unique Spike mutation sets to ``lineage_spike.fasta``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = output_dir / LINEAGE_SPIKE_FASTA
    n_records = 0

    with open(fasta_path, "w", encoding="utf-8") as fasta_out:
        for spike_key, lineages in sorted(groups.items(), key=lambda kv: kv[1]):
            header = lineage_group_header(lineages)
            mut_seq = apply_mutations_flat_list(list(spike_key), ref_spike_aa)
            fasta_out.write(f">{header}\n")
            fasta_out.write(mut_seq + "\n")
            n_records += 1

    return n_records


def parse_args() -> argparse.Namespace:
    default_ref_spike = _repo_root / "data" / "ref_spike_genome.fa"

    p = argparse.ArgumentParser(
        description=(
            "DFS lineage roots on a MAT → Spike AA mutations → merge identical sets "
            "→ lineage_spike.fasta."
        )
    )
    p.add_argument("--mat", required=True, help="Mutation Annotated Tree (.pb / .pb.gz).")
    p.add_argument(
        "--reference",
        required=True,
        help="Full SARS-CoV-2 reference FASTA (29903 bp) for nt→AA translation.",
    )
    p.add_argument(
        "--ref-spike",
        default=str(default_ref_spike),
        help=f"Spike amino-acid reference FASTA (default: {default_ref_spike}).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for lineage_spike.fasta.",
    )
    p.add_argument(
        "--lineages",
        default=None,
        help="Optional comma-separated lineage filter (default: all lineage roots in tree).",
    )
    p.add_argument(
        "--lineages-file",
        default=None,
        metavar="PATH",
        help="Optional text file: one lineage per line (# comments allowed).",
    )
    p.add_argument(
        "--annotation-index",
        type=int,
        default=1,
        help="Index into node.annotations for the lineage label (default: 1).",
    )
    return p.parse_args()


def load_wanted_lineages(args: argparse.Namespace) -> Optional[Set[str]]:
    if not args.lineages and not args.lineages_file:
        return None

    wanted: Set[str] = set()
    if args.lineages:
        wanted.update(s.strip() for s in args.lineages.split(",") if s.strip())
    if args.lineages_file:
        with open(args.lineages_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                wanted.add(line)
    return wanted


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.reference):
        raise FileNotFoundError(f"Reference not found: {args.reference}")
    if not os.path.isfile(args.ref_spike):
        raise FileNotFoundError(f"Spike reference not found: {args.ref_spike}")
    if not os.path.isfile(args.mat):
        raise FileNotFoundError(f"MAT not found: {args.mat}")

    wanted = load_wanted_lineages(args)
    output_dir = Path(args.output_dir).expanduser().resolve()

    print("Loading full-genome reference for translation...")
    refseq = gaam.load_reference_sequence_modified(args.reference)

    print(f"Loading Spike reference from {args.ref_spike} ...")
    ref_spike_aa = str(getReference(args.ref_spike))

    print(f"Loading MAT: {args.mat}")
    tree = bte.MATree(args.mat)

    print("Collecting lineage roots (depth-first expansion)...")
    rows = collect_lineage_root_rows(
        tree,
        refseq,
        annotation_index=args.annotation_index,
        wanted=wanted,
    )
    print(f"Found {len(rows)} lineage root(s).")

    if wanted is not None:
        found = {r["lineage"] for r in rows}
        missing = wanted - found
        if missing:
            print(
                "Warning: no lineage root found for: "
                + ", ".join(sorted(missing)),
                file=sys.stderr,
            )

    groups = merge_lineages_by_spike_mutations(rows)
    print(
        f"Merged into {len(groups)} unique Spike mutation set(s) "
        f"({sum(len(v) for v in groups.values())} lineage assignments)."
    )

    n_records = write_lineage_spike_fasta(groups, ref_spike_aa, output_dir)
    fasta_path = output_dir / LINEAGE_SPIKE_FASTA
    print(f"Wrote {n_records} sequence(s) to {fasta_path}")


if __name__ == "__main__":
    main()
