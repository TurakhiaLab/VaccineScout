"""
Script to extract single-nucleotide mutations from a MAT, translate them to
amino acid mutations, keep only Spike (``S:...``) calls, and write one Parquet file
with every tree node (internal nodes and leaf haplotypes).

First runs ``scripts/get_mutations`` (must be built with BUILD_GET_MUTATIONS) to write
``{output-dir}/{output-stem}_nucleotide.parquet`` derived from ``--output-mutations``,
then translates those nt haplotypes to amino acid space.

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
import subprocess
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
    spike_nt = refseq[start_1b - 1 : end_1b]
    spike_aa = translate_dna_to_aa(spike_nt)
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(
            f">S|reference|protein|len={len(spike_aa)}|cds_nt={start_1b}-{end_1b}\n"
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


def write_aa_mutations_parquet_from_nt_parquet(
    refseq: str,
    filename: str,
    nt_parquet_path: str,
    batch_size: int = 100_000,
    progress_every: Optional[int] = 100_000,
    progress_label: str = "nodes",
) -> None:
    """
    Read ``node_name`` (or ``sample_id``) and ``nt_mutations`` (;-separated nt
    strings) from Parquet — e.g. from ``get_mutations`` — translate to Spike-only
    amino acid mutations, and write (``sample_id``, ``aa_mutations``) Parquet
    using the same schema as :func:`write_aa_mutations_parquet`.
    """
    schema = pa.schema(
        [
            ("sample_id", pa.large_string()),
            ("aa_mutations", pa.large_string()),
        ]
    )
    pf = pq.ParquetFile(nt_parquet_path)
    col_names = pf.schema.names
    if "node_name" in col_names:
        node_col = "node_name"
    elif "sample_id" in col_names:
        node_col = "sample_id"
    else:
        raise ValueError(
            "nt-mutations Parquet must have column 'node_name' or 'sample_id'; "
            f"got {col_names!r}"
        )
    if "nt_mutations" not in col_names:
        raise ValueError(
            f"nt-mutations Parquet must have column 'nt_mutations'; got {col_names!r}"
        )

    buf_ids: List[str] = []
    buf_aa: List[str] = []
    processed = 0

    with pq.ParquetWriter(filename, schema) as writer:
        for batch in pf.iter_batches(
            batch_size=batch_size, columns=[node_col, "nt_mutations"]
        ):
            nodes = batch.column(0).to_pylist()
            nt_raw_list = batch.column(1).to_pylist()
            for sample, nt_raw in zip(nodes, nt_raw_list):
                if nt_raw is None:
                    nt_str = ""
                else:
                    nt_str = str(nt_raw)
                nt_list = [p.strip() for p in nt_str.split(";") if p.strip()]
                aa_mutations = nuc_mutations_to_aa_mutations_modified(
                    refseq, nt_list
                )
                spike_only = [m for m in aa_mutations if m.startswith("S:")]
                buf_ids.append(sample)
                buf_aa.append(";".join(spike_only))

                if len(buf_ids) >= batch_size:
                    writer.write_table(
                        pa.Table.from_arrays(
                            [pa.array(buf_ids), pa.array(buf_aa)],
                            schema=schema,
                        )
                    )
                    buf_ids.clear()
                    buf_aa.clear()

                processed += 1
                if progress_every and processed % progress_every == 0:
                    print(f"{processed} {progress_label} processed.")

        if buf_ids:
            writer.write_table(
                pa.Table.from_arrays(
                    [pa.array(buf_ids), pa.array(buf_aa)],
                    schema=schema,
                )
            )


def write_aa_mutations_parquet(
    tree,
    refseq,
    filename,
    node_ids: Sequence[str],
    batch_size: int = 100_000,
    progress_every: Optional[int] = 100_000,
    progress_label: str = "nodes",
):
    """
    For each node ID in ``node_ids``, translate nucleotide mutations to amino acid
    space and write only Spike gene mutations (strings like ``S:G639V``) to Parquet.
    Rows are flushed in batches of ``batch_size`` to limit memory use.
    Progress is printed every ``progress_every`` rows; set to ``0`` or ``None`` to disable.
    """
    schema = pa.schema(
        [
            ("sample_id", pa.large_string()),
            ("aa_mutations", pa.large_string()),
        ]
    )
    buf_ids: List[str] = []
    buf_aa: List[str] = []

    with pq.ParquetWriter(filename, schema) as writer:
        for i, sample in enumerate(node_ids):
            nt_mutations = list(tree.get_haplotype(sample))
            aa_mutations = nuc_mutations_to_aa_mutations_modified(
                refseq, nt_mutations
            )
            spike_only = [m for m in aa_mutations if m.startswith("S:")]
            buf_ids.append(sample)
            buf_aa.append(";".join(spike_only))

            if len(buf_ids) >= batch_size:
                writer.write_table(
                    pa.Table.from_arrays(
                        [pa.array(buf_ids), pa.array(buf_aa)],
                        schema=schema,
                    )
                )
                buf_ids.clear()
                buf_aa.clear()

            if progress_every and (i + 1) % progress_every == 0:
                print(f"{i + 1} {progress_label} processed.")

        if buf_ids:
            writer.write_table(
                pa.Table.from_arrays(
                    [pa.array(buf_ids), pa.array(buf_aa)],
                    schema=schema,
                )
            )


def nt_mutations_parquet_path(output_mutations: str) -> str:
    """
    From ``--output-mutations`` (e.g. ``.../output.parquet``), return the path for
    nucleotide haplotypes: same directory, ``{stem}_nucleotide.parquet``.
    """
    p = Path(output_mutations).expanduser().resolve()
    return str(p.parent / f"{p.stem}_nucleotide{p.suffix}")


def run_get_mutations(mat_path: str, nt_parquet_out: str) -> None:
    """Invoke ``scripts/get_mutations`` built next to this file."""
    script_dir = Path(__file__).resolve().parent
    exe = script_dir / ("get_mutations.exe" if os.name == "nt" else "get_mutations")
    if not exe.is_file():
        raise FileNotFoundError(
            f"Missing '{exe}'. Build it (e.g. cmake -DBUILD_GET_MUTATIONS=ON) "
            "and ensure the binary sits alongside this script."
        )
    os.makedirs(os.path.dirname(os.path.abspath(nt_parquet_out)), exist_ok=True)
    cmd = [str(exe), "--mat", mat_path, "--output", nt_parquet_out]
    print(f"Running {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


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
            "mutations (Spike-only, e.g. S:G639V), and write one Parquet file covering every "
            "node (internal + leaves)."
        )
    )
    parser.add_argument(
        "-o",
        "--output-mutations",
        required=True,
        dest="output_mutations",
        help="Parquet path: one row per tree node (sample_id, aa_mutations Spike-only).",
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
    print(
        f"Writing parent-child nodes to {output_parent_child} ..."
    )
    write_tree_parent_child_csv(tree, output_parent_child)

    nt_parquet = nt_mutations_parquet_path(args.output_mutations)
    run_get_mutations(args.mat, nt_parquet)
    print(f"Wrote nucleotide mutations Parquet to {nt_parquet}")

    write_aa_mutations_parquet_from_nt_parquet(
        refseq, args.output_mutations, nt_parquet
    )

    print(
        "Process complete! Wrote "
        f"{nt_parquet}, {args.output_mutations}, and {output_parent_child}"
    )


if __name__ == "__main__":
    main()
