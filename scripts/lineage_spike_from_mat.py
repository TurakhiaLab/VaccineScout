#!/usr/bin/env python3
"""
For each requested Pango/lineage label, find an internal MAT node whose UShER
``clade_annotations`` entry at index 1 matches (typical Pango lineage field),
then collect cumulative Spike AA mutations and the mutated Spike protein sequence.

Uses the same reference genome + nucleotide→AA translation path as
``get_amino_acid_mutations_of_sequences.py``, and ``Helpers.apply_mutations_flat_list``
for the amino-acid sequence (same idea as ``create_from_metadata.ipynb``).

Requires: ``bte``, local ``pyro-cov`` checkout (see get_amino_acid_mutations_of_sequences.py),
``pandas``.

Example:
    python scripts/lineage_spike_from_mat.py \\
        --mat /path/to/lineageTree.pb \\
        --reference /path/to/ref.fa \\
        --lineages B.1,B.1.1.7,XBB.1.5 \\
        --output vaccine_lineages.pkl
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Set

import bte
import pandas as pd

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

# pyro-cov path + translation helpers (loads pyrocov from ./pyro-cov)
import get_amino_acid_mutations_of_sequences as gaam  # noqa: E402

from Helpers import apply_mutations_flat_list, get_mutation_pattern  # noqa: E402


def ref_spike_amino_acids(refseq: str) -> str:
    start_1b, end_1b = gaam.GENE_TO_POSITION["S"]
    spike_nt = refseq[start_1b - 1 : end_1b - 3]  # omit terminal stop codon
    return gaam.translate_dna_to_aa(spike_nt)


def sort_spike_mutations(spike_muts: Sequence[str]) -> List[str]:
    """Apply mutations in genomic order within Spike to match a single final haplotype."""

    def site_key(m: str) -> int:
        _o, site, _n = get_mutation_pattern(m)
        return site if site else 0

    return sorted([m for m in spike_muts if m], key=site_key)


def lineage_from_node(node, annotation_index: int) -> Optional[str]:
    ann = list(node.annotations)
    if len(ann) <= annotation_index:
        return None
    s = ann[annotation_index].strip()
    return s or None


def collect_lineage_spike_rows(
    tree: bte.MATree,
    refseq: str,
    wanted: Set[str],
    *,
    annotation_index: int = 1,
    first_only: bool = True,
    include_node_id: bool = False,
) -> List[dict]:
    """
    DFS (``depth_first_expansion``): internal nodes only; match
    ``node.annotations[annotation_index]`` against ``wanted``.
    """
    ref_spike_aa = ref_spike_amino_acids(refseq)
    wanted_left = set(wanted)
    rows: List[dict] = []

    for node in tree.depth_first_expansion():
        if node.is_leaf():
            continue
        lineage = lineage_from_node(node, annotation_index)
        if lineage is None or lineage not in wanted_left:
            continue

        nt_path = list(tree.get_haplotype(node.id))
        aa_path = gaam.nuc_mutations_to_aa_mutations_modified(refseq, nt_path)
        spike_muts = [m for m in aa_path if m.startswith("S:")]
        spike_ordered = sort_spike_mutations(spike_muts)
        mut_seq = apply_mutations_flat_list(spike_ordered, ref_spike_aa)

        row = {
            "clade": lineage,
            "parsed_spike_list": spike_muts,
            "mutated_sequence": mut_seq,
        }
        if include_node_id:
            row["node_id"] = node.id
        rows.append(row)

        if first_only:
            wanted_left.discard(lineage)
            if not wanted_left:
                break

    missing = wanted_left
    if missing:
        print(
            "Warning: no matching internal node found for lineages: "
            + ", ".join(sorted(missing)),
            file=sys.stderr,
        )

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export clade / Spike mutation list / mutated Spike sequence for "
            "lineages defined on internal MAT nodes (clade_annotations)."
        )
    )
    p.add_argument("--mat", required=True, help="Mutation Annotated Tree (.pb / .pb.gz).")
    p.add_argument(
        "--reference",
        required=True,
        help="SARS-CoV-2 reference FASTA (29903 bp, same as get_amino_acid_mutations_of_sequences.py).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output path for pickle (pandas DataFrame).",
    )
    p.add_argument(
        "--lineages",
        default=None,
        help="Comma-separated lineage names (must match annotations[INDEX] exactly).",
    )
    p.add_argument(
        "--lineages-file",
        default=None,
        metavar="PATH",
        help="Text file: one lineage per line (use # for comments).",
    )
    p.add_argument(
        "--annotation-index",
        type=int,
        default=1,
        help=(
            "Index into node.annotations / clade_annotations for the lineage label "
            "(default: 1, typical Pango field on dual-annotated UShER trees)."
        ),
    )
    p.add_argument(
        "--all-matches",
        action="store_true",
        help=(
            "Emit every matching internal node; by default only the first DFS hit "
            "per requested lineage is kept."
        ),
    )
    p.add_argument(
        "--with-node-id",
        action="store_true",
        help="Add node_id column (MAT internal node name).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.lineages and not args.lineages_file:
        print("Provide --lineages and/or --lineages-file.", file=sys.stderr)
        sys.exit(2)

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

    if not os.path.isfile(args.reference):
        raise FileNotFoundError(f"Reference not found: {args.reference}")
    if not os.path.isfile(args.mat):
        raise FileNotFoundError(f"MAT not found: {args.mat}")

    refseq = gaam.load_reference_sequence_modified(args.reference)
    print(f"Loading MAT: {args.mat}")
    tree = bte.MATree(args.mat)

    rows = collect_lineage_spike_rows(
        tree,
        refseq,
        wanted,
        annotation_index=args.annotation_index,
        first_only=not args.all_matches,
        include_node_id=args.with_node_id,
    )

    df = pd.DataFrame(rows)
    out = os.path.abspath(args.output)
    df.to_pickle(out)
    print(f"Wrote {len(df)} row(s) to {out}")


if __name__ == "__main__":
    main()
