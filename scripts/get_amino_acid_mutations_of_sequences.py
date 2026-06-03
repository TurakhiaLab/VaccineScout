"""
Script to extract single-nucleotide mutations from a MAT via ``bte``,
translate them to amino acid mutations, keep only Spike (``S:...``) calls,
and write Parquet files with every tree node (internal nodes and leaf haplotypes).

Writes two files next to ``--output-mutations``:

- ``<stem>.parquet`` — cumulative haplotype via ``tree.get_haplotype(node_id)``
  (mutations from reference along the path to the node).
- ``<stem>_branch.parquet`` — mutations stored only on that node's incoming branch
  (``node.mutations``), not the full ancestral path.

Usage:
    python scripts/get_amino_acid_mutations_of_sequences.py \\
        --output-mutations <output.parquet> \\
        --output-parent-child <edges.csv> \\
        --mat <path/to/tree.pb[.gz]> \\
        --reference <path/to/reference.fasta>

This script expects the `pyro-cov` repository to be checked out in the current
working directory so that the `pyrocov` package can be imported from it.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Sequence

import bte
import pyarrow as pa
import pyarrow.parquet as pq

REF_SPIKE_GENOME_FILENAME = "ref_spike_genome.fa"


def _ensure_pyrocov_on_path():
    """Add the local `pyro-cov` checkout to sys.path so `pyrocov` can be imported."""
    pyrocov_root = os.path.join(os.getcwd(), "pyro-cov")
    if not os.path.isdir(os.path.join(pyrocov_root, "pyrocov")):
        raise FileNotFoundError(
            f"Could not find the 'pyrocov' package under '{pyrocov_root}'. "
            "Please clone https://github.com/broadinstitute/pyro-cov into the "
            "current working directory."
        )
    if pyrocov_root not in sys.path:
        sys.path.insert(0, pyrocov_root)


_ensure_pyrocov_on_path()

# Imported after sys.path is patched so that the local `pyro-cov` checkout is used.
from pyrocov.sarscov2 import GENE_TO_POSITION, DNA_TO_AA  # noqa: E402

# ---------------------------------------------------------------------------
# Adapted from:
# https://github.com/TurakhiaLab/SARS2_RecombinationAnalysis/blob/main/notebooks/third_party/nuc_mutations_to_aa_mutations_modified.py
# Originally from:
# https://github.com/broadinstitute/pyro-cov/blob/9f84acc3ddff9bcb55c8d1b77fd23204a4f54b8e/pyrocov/sarscov2.py
# Licensed under the Apache License, Version 2.0.
# ---------------------------------------------------------------------------
def load_reference_sequence_modified(reference_path: str) -> str:
    """Load a SARS-CoV-2 reference fasta from an arbitrary path."""
    with open(reference_path) as f:
        ref = "".join(line.strip() for line in f if not line.startswith(">"))
    assert len(ref) == 29903, len(ref)
    return ref


def translate_dna_to_aa(nt: str) -> str:
    """Translate a coding sequence in triplets using pyrocov's ``DNA_TO_AA``."""
    assert len(nt) % 3 == 0, len(nt)
    out = []
    for i in range(0, len(nt), 3):
        codon = nt[i : i + 3]
        aa = DNA_TO_AA[codon]
        out.append("*" if aa is None else aa)
    return "".join(out)


def write_reference_spike_genome_fasta(refseq: str, output_path: str) -> str:
    """
    Write the reference S (spike) protein sequence to a FASTA file using
    pyrocov's GENE_TO_POSITION and ``DNA_TO_AA`` (1-based Wuhan-Hu-1 coords for
    the underlying CDS).

    ``output_path`` is the full path for the FASTA file. ``main()`` writes it in
    the same directory as the Parquet path given by ``--output-mutations``.
    """
    start_1b, end_1b = GENE_TO_POSITION["S"]
    # GFF end is inclusive and covers the stop codon; omit last 3 nt for protein.
    cds_end_1b = end_1b - 3
    spike_nt = refseq[start_1b - 1 : cds_end_1b]
    spike_aa = translate_dna_to_aa(spike_nt)
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(
            f">S|reference|protein|len={len(spike_aa)}|cds_nt={start_1b}-{cds_end_1b}\n"
        )
        out.write(spike_aa + "\n")
    return output_path


def nuc_mutations_to_aa_mutations_modified(refseq: str, ms: List[str]) -> List[str]:
    """Translate a list of nucleotide mutations to amino acid mutations."""
    REFERENCE_SEQ = refseq
    ms_by_aa = defaultdict(list)

    for m in ms:
        if isinstance(m, str):
            position_nuc = int(m[1:-1])
            new_nuc = m[-1]
        else:
            position_nuc = m.position
            new_nuc = m.mut

        for gene, (start, end) in GENE_TO_POSITION.items():
            if start <= position_nuc <= end:
                position_aa = (position_nuc - start) // 3
                position_codon = (position_nuc - start) % 3
                ms_by_aa[gene, position_aa].append((position_codon, new_nuc))

    result = []
    for (gene, position_aa), codon_mods in ms_by_aa.items():
        start, _end = GENE_TO_POSITION[gene]
        pos = start + position_aa * 3
        pos -= 1  # convert from 1-based to 0-based
        old_codon = REFERENCE_SEQ[pos : pos + 3]
        new_codon = list(old_codon)
        for position_codon, new_nuc in codon_mods:
            new_codon[position_codon] = new_nuc
        new_codon = "".join(new_codon)

        old_aa = DNA_TO_AA[old_codon]
        new_aa = DNA_TO_AA[new_codon]
        if new_aa == old_aa:  # ignore synonymous substitutions
            continue
        if old_aa is None:
            old_aa = "STOP"
        if new_aa is None:
            new_aa = "STOP"
        result.append(f"{gene}:{old_aa}{position_aa + 1}{new_aa}")  # 1-based
    return result


def branch_aa_mutations_parquet_path(output_mutations: str) -> str:
    """``output.parquet`` → ``output_branch.parquet`` (same directory, stem + ``_branch``)."""
    p = Path(output_mutations).expanduser().resolve()
    return str(p.parent / f"{p.stem}_branch{p.suffix}")


def write_cumulative_and_branch_aa_mutations_parquets(
    tree,
    refseq,
    cumulative_filename: str,
    branch_filename: str,
    nodes: Sequence,
    batch_size: int = 100_000,
    progress_every: Optional[int] = 100_000,
    progress_label: str = "nodes",
) -> None:
    """
    For each MAT node in preorder ``nodes``, write two Parquet files (same schema):

    - Cumulative Spike AA mutations from ``tree.get_haplotype(node.id)``.
    - Branch-only Spike AA mutations from ``node.mutations`` (incoming edge).
    """
    schema = pa.schema(
        [
            ("sample_id", pa.large_string()),
            ("aa_mutations", pa.large_string()),
        ]
    )
    buf_c_ids: List[str] = []
    buf_c_aa: List[str] = []
    buf_b_ids: List[str] = []
    buf_b_aa: List[str] = []

    def flush_pair(w_c, w_b) -> None:
        if buf_c_ids:
            w_c.write_table(
                pa.Table.from_arrays(
                    [pa.array(buf_c_ids), pa.array(buf_c_aa)],
                    schema=schema,
                )
            )
            buf_c_ids.clear()
            buf_c_aa.clear()
        if buf_b_ids:
            w_b.write_table(
                pa.Table.from_arrays(
                    [pa.array(buf_b_ids), pa.array(buf_b_aa)],
                    schema=schema,
                )
            )
            buf_b_ids.clear()
            buf_b_aa.clear()

    with pq.ParquetWriter(cumulative_filename, schema) as w_cumulative, pq.ParquetWriter(
        branch_filename, schema
    ) as w_branch:
        for i, node in enumerate(nodes):
            sample = node.id
            nt_path = list(tree.get_haplotype(sample))
            nt_branch = list(node.mutations)

            aa_path = nuc_mutations_to_aa_mutations_modified(refseq, nt_path)
            aa_branch = nuc_mutations_to_aa_mutations_modified(refseq, nt_branch)

            spike_path = [m for m in aa_path if m.startswith("S:")]
            spike_branch = [m for m in aa_branch if m.startswith("S:")]

            buf_c_ids.append(sample)
            buf_c_aa.append(";".join(spike_path))
            buf_b_ids.append(sample)
            buf_b_aa.append(";".join(spike_branch))

            if len(buf_c_ids) >= batch_size:
                flush_pair(w_cumulative, w_branch)

            if progress_every and (i + 1) % progress_every == 0:
                print(f"{i + 1} {progress_label} processed.")

        flush_pair(w_cumulative, w_branch)


def write_tree_parent_child_csv(tree, csv_path: str) -> None:
    """
    Write one row per node from preorder ``depth_first_expansion``: parent ID,
    then node ID. Root rows use the literal ``NULL`` for the parent column.
    """
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["parent_node", "node"])
        for node in tree.depth_first_expansion():
            parent = node.parent
            parent_name = "NULL" if parent is None else parent.id
            writer.writerow([parent_name, node.id])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract single-nucleotide mutations from a MAT, translate them to amino acid "
            "mutations (Spike-only, e.g. S:G639V), and write Parquet files covering every "
            "node (internal + leaves): cumulative haplotype and branch-only mutations."
        )
    )
    parser.add_argument(
        "-o",
        "--output-mutations",
        required=True,
        dest="output_mutations",
        help=(
            "Parquet path for cumulative haplotype (sample_id, aa_mutations Spike-only). "
            "Also writes <stem>_branch<suffix> for mutations on each node's branch only."
        ),
    )
    parser.add_argument(
        "--output-parent-child",
        required=True,
        metavar="PATH",
        dest="output_parent_child",
        help=(
            "CSV path: one row per tree node (parent_node, node) from depth_first_expansion; "
            "root uses NULL as parent_node."
        ),
    )
    parser.add_argument(
        "--mat",
        required=True,
        help="Path to the Mutation Annotated Tree (MAT) protobuf file.",
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to the SARS-CoV-2 reference fasta (29903 bp).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.reference):
        raise FileNotFoundError(f"Reference fasta not found: '{args.reference}'")
    if not os.path.isfile(args.mat):
        raise FileNotFoundError(f"MAT file not found: '{args.mat}'")

    print("Loading Reference Sequence...")
    refseq = load_reference_sequence_modified(args.reference)

    output_dir = os.path.dirname(os.path.abspath(args.output_mutations))
    spike_fa = os.path.join(output_dir, REF_SPIKE_GENOME_FILENAME)
    spike_fa = write_reference_spike_genome_fasta(refseq, spike_fa)
    print(f"Wrote reference spike nucleotide FASTA to {spike_fa}")

    print(f"Loading MAT file: {args.mat}")
    tree = bte.MATree(args.mat)

    output_parent_child = os.path.abspath(args.output_parent_child)
    print(f"Writing parent-child nodes to {output_parent_child} ...")
    write_tree_parent_child_csv(tree, output_parent_child)

    nodes = list(tree.depth_first_expansion())
    branch_parquet = branch_aa_mutations_parquet_path(args.output_mutations)
    print(
        f"Writing cumulative haplotype Parquet to {args.output_mutations} "
        f"and branch mutations to {branch_parquet} ..."
    )
    write_cumulative_and_branch_aa_mutations_parquets(
        tree,
        refseq,
        args.output_mutations,
        branch_parquet,
        nodes,
    )

    print(
        "Process complete! Wrote "
        f"{args.output_mutations}, {branch_parquet}, and {output_parent_child}"
    )


if __name__ == "__main__":
    main()
