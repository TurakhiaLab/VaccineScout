<div align="center">

# VaccineScout

[license-badge]: https://img.shields.io/badge/License-MIT-yellow.svg 
[license-link]: https://github.com/TurakhiaLab/VaccineScout/blob/main/LICENSE

[![License][license-badge]][license-link]

<div align="center">
  <img src="docs/images/VaccineScout_logo.svg" width="300"/>
</div>

</div>

## Table of Contents
- [Introduction](#intro)
- [Installation](#install)
  - [1. Clone the repositories and install the dependencies](#clone)
  - [2. Create the environment](#envs)
  - [3. Build the C++ helpers](#build)
  - [4. Register the Jupyter kernel](#kernel)
- [Configuration](#config)
- [Running the Pipeline](#pipeline)
- [Repository Layout](#layout)
- [Troubleshooting](#troubleshooting)
- [Citing Scout](#cite)

<br>

## <a name="intro"></a> Introduction

VaccineScout is a computational framework that uses genomic surveillance data
and fitness estimates to project the future prevalence of circulating SARS-CoV-2
variants. It then selects varying number of medoids that minimize the forecasted-
prevalence-weighted spike sequence distance to these and generates a distance-
versus-valency curve to select the optimal vaccine valency.

<div align="center">
   <img src="docs/images/VaccineScout_overview.svg" style="display: block; margin-left: auto; margin-right: auto; width: 100%;">
  <div> <b>Figure 1: Overview of VaccineScout.</b> </div>
</div>

## <a name="install"></a> Installation

### <a name="clone"></a> 1. Clone repositories and install the dependencies
Clone the VaccineScout repository, then set up the PyR0 and CoVFit.

```bash
# i. VaccineScout — this repository
git clone https://github.com/TurakhiaLab/VaccineScout.git
cd VaccineScout
```

```bash
# ii. pyro-cov — nucleotide -> amino-acid translation tables
git clone https://github.com/broadinstitute/pyro-cov
```

```bash
# iii. CoVFit CLI — prebuilt GPU fitness model
curl -L -o covfit_cli_20241007.tar.gz \
  "https://zenodo.org/records/14438178/files/covfit_cli_20241007.tar.gz?download=1"
tar -xzf covfit_cli_20241007.tar.gz
rm covfit_cli_20241007.tar.gz
```

⚠️ You can unzip CoVFit elsewhere, make sure to point `paths.covfit_cli` in 
`config.yaml` at that directory instead.

### <a name="envs"></a> 2. Create the environment

```bash
conda env create -f envs/vaccinescout.yml
```

[Chronumental](https://github.com/theosanderson/chronumental) infers a date for
every node on the tree. You need it to regenerate `files.chronumental_dates`.
You can skip this step if you want to infer haplotype dates from their names.

```bash
conda activate vaccinescout

# Required: the jaxlib wheel dlopen()s cuDNN at run time, and conda's cuDNN 8.9
# is a CUDA 11 build, so it needs cudatoolkit beside it.
conda install -y -c conda-forge cudatoolkit=11.8.0 "cudnn<9.0"
conda install -y -c bioconda chronumental

# conda pulls a CPU jaxlib. Remove both jax packages before installing the CUDA
# --force keeps chronumental itself.
conda remove --force -y jax jaxlib

pip install "jax==0.4.26" "jaxlib==0.4.26+cuda12.cudnn89" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### <a name="build"></a> 3. Build the C++ helpers

`min_dist` and `pairwise_dist` compute Hamming distances over the sequences
and are invoked directly by the notebooks. The binaries are written to 
`scripts/min_dist` and `scripts/pairwise_dist`.

```bash
conda activate vaccinescout
cmake -S scripts -B scripts/build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_C_COMPILER="$CC"
cmake --build scripts/build -j
```

### <a name="kernel"></a> 4. Register the Jupyter kernel

```bash
conda activate vaccinescout
python -m ipykernel install --user \
  --name vaccinescout --display-name "Python (vaccinescout)"
```

## <a name="config"></a> Configuration

Every path and run parameter lives in **`config.yaml`** at the repository root.
No script or notebook hardcodes a location; they all read this file through
`scripts/config_loader.py`.

Print the resolved configuration and check which inputs are present:

```bash
python scripts/config_loader.py
```

Run it first whenever something cannot be found. Output is split in two, because
`MISSING` means very different things in each half:

* **CRITICAL** — inputs the pipeline reads: the data root and third-party 
  checkouts, the reference genome, the MAT/Newick/metadata files, the
  PyR<sub>0</sub> and EVEscape tables, the Pfizer FASTA, and the 
  `covfit_cli` / `min_dist` / `pairwise_dist` executables. Anything
  MISSING here is a hard stop — download, clone or build it.
* **NON-CRITICAL** — intermediates, each labelled with the step that writes it
  and listed in pipeline order. MISSING just means that step has not run yet,
  which is the normal state of a fresh checkout.

Two distinct references are easy to confuse:
* `files.reference` — the full **genome** (29903 nt), used for
  nucleotide→amino-acid translation.
* `files.ref_spike_genome` — the 1273 aa Spike **protein** FASTA, used as the
  backbone that mutations are applied to.

## <a name="pipeline"></a> Running the Pipeline

Run the ten steps in this order. All of them run in the single
`vaccinescout` environment. Steps 5a and 5b are alternatives — run one, not both;
they build the same three files from two different sources of dates.

| # | Step | Reads | Writes | 
| --- | --- | --- | --- |
| 0 | `chronumental` | `files.tree_newick`, `files.metadata` | `files.chronumental_dates` |
| 1 | `get_amino_acid_mutations_of_sequences.py` | `files.mat`, `files.reference` | `files.node_spike_cumulative`, `…_branch`, `files.parent_node`, `ref_spike_genome.fa` |
| 2 | `get_lineage_fasta.py` | `files.mat`, `files.reference`, `files.ref_spike_genome` | `files.lineage_spike_fasta` |
| 3 | `lineage_spike_from_mat.py` | `files.mat`, `files.reference`, `mat_scripts.lineages` | `files.fda_recommendations` |
| 4 | `pyro_aug.py` | PyR<sub>0</sub> `mutations.tsv` | PyR<sub>0</sub> Spike pickle (`fitness.pyro_spike_filename`) |
| 5a | `create_from_chronumental_data.ipynb` | `files.node_spike_cumulative`, `files.chronumental_dates` | `files.final_df`, `files.sequence_dict`, `files.mutation_path_dict` |
| 5b | `create_from_metadata.ipynb` *(alternative to 5a — dates from haplotype names)* | `files.node_spike_cumulative` | same three files as 5a |
| 6 | `rank_analysis.ipynb` | `final_df`, `sequence_dict`, PyR<sub>0</sub>, EVEscape, CoVFit | `<cutoff>/overlap.svg` |
| 7 | `new_mutations_cumulative.ipynb` | `final_df`, `sequence_dict`, `min_dist` | `<cutoff>/dist_*.csv`, `new_mutations.svg` |
| 8 | `new_mutations_fitness.ipynb` | `final_df`, `parent_node`, `node_spike_branch` | `<cutoff>/fitness.svg` |
| 9 | `scout_selection.ipynb` | everything above, `pairwise_dist`, CoVFit | `scout_medoid_report.*`, medoid plots |


### Steps 0–4 — tree dating, MAT processing and the PyR<sub>0</sub> table

All five run from one driver script, which reads `config.yaml`:

```bash
conda activate vaccinescout
./scripts/run_mat_pipeline.sh            # steps 0-4
```

Every step runs every time and overwrites its outputs; nothing is skipped because
a file happens to already be there. Comment out the steps you do not want to
repeat, or run them individually as documented below.

The rest of this section documents each step individually, for when you want to
run one on its own or override a path.

**Step 0 — Date every node on the tree** with
[Chronumental](https://github.com/theosanderson/chronumental). Step 5a needs this
result, while its alternate, step 5b does not.

Arguments:
| Flag | Config default | Meaning |
| --- | --- | --- |
| `--tree` | `files.tree_newick` | Newick tree matching `files.mat`. |
| `--dates` | `files.metadata` | Metadata TSV with `strain` and `date` columns. |
| `--reference_node` | `chronumental.reference_node` | Node used to anchor the time axis. |
| `--dates_out` | `files.chronumental_dates` | Where the dated output lands. |
| `--steps` | `chronumental.steps` | Optimisation steps (2000). |
| `--use_gpu` | `chronumental.use_gpu` | Requires the GPU JAX build. |

⚠️ You can skip this step if you want to take haplotype dates from their names
instead. Set `chronumental.run: false` in `config.yaml` and run **step 5b** 
(`create_from_metadata.ipynb`) rather than step 5a. But it will prevent you 
from running `Step 8 - new_mutations_fitness` that requires dates for internal nodes.

**Step 1 — Spike amino-acid mutations for every node.**
Walks the tree in depth-first order and writes, per node, the cumulative mutations
for the haplotype and the branch-only mutations, plus the parent<->child list. Also
writes the reference Spike protein FASTA next to the mutations Parquet.

```bash
python scripts/get_amino_acid_mutations_of_sequences.py
```

Arguments:
| Flag | Config default | Meaning |
| --- | --- | --- |
| `--mat` | `files.mat` | Mutation-annotated tree (`.pb` / `.pb.gz`). |
| `--reference` | `files.reference` | Full 29903 bp reference **genome**. |
| `-o`, `--output-mutations` | `files.node_spike_cumulative` | Cumulative-haplotype Parquet. `<stem>_branch<suffix>` is written alongside. |
| `--output-parent-child` | `files.parent_node` | Edge-list CSV; the root's parent is the literal `NULL`. |

**Step 2 — Spike sequence per lineage.**
Finds each lineage root, translates its cumulative haplotype to Spike AA calls,
merges lineages sharing an identical mutation set, and writes them as a single
FASTA (headers join merged lineages with `|`).

```bash
python scripts/get_lineage_fasta.py
```

Arguments:
| Flag | Config default | Meaning |
| --- | --- | --- |
| `--mat` | `files.mat` | Mutation-annotated tree. |
| `--reference` | `files.reference` | Full reference **genome**, for nt→AA translation. |
| `--ref-spike` | `files.ref_spike_genome` | Spike **protein** FASTA (1273 aa) that mutations are applied to — written by pipeline step 1. |
| `--output-dir` | directory of `files.lineage_spike_fasta` | Destination directory; the file is always named `lineage_spike.fasta`. |
| `--lineages` | *(none — all lineage roots)* | Optional comma-separated filter. |
| `--lineages-file` | *(none)* | Optional file, one lineage per line (`#` comments allowed). |
| `--annotation-index` | `mat_scripts.annotation_index` | Which `node.annotations` entry holds the Pango label. |

**Step 3 — Spike sequences for the vaccine reference strains.**
Same translation, restricted to the lineages you want to compare VaccineScout's selections
against, saved as a pandas pickle.

```bash
python scripts/lineage_spike_from_mat.py
```

Arguments:
| Flag | Config default | Meaning |
| --- | --- | --- |
| `--mat` | `files.mat` | Mutation-annotated tree. |
| `--reference` | `files.reference` | Full reference **genome**. |
| `--output` | `files.fda_recommendations` | Pickled DataFrame: `clade`, `parsed_spike_list`, `mutated_sequence`. |
| `--lineages` | `mat_scripts.lineages` | Comma-separated lineage names. |
| `--lineages-file` | *(none)* | Alternative to `--lineages`. |
| `--annotation-index` | `mat_scripts.annotation_index` | Pango label index (`1` on dual-annotated UShER trees). |
| `--all-matches` | off | Emit every matching node, not just the first DFS hit per lineage. |
| `--with-node-id` | off | Add the MAT internal node name as a column. |

**Step 4 — PyR<sub>0</sub> Spike score table.**
The PyR<sub>0</sub> release only scores the substitutions it observed, but the
later notebooks need a score for any substitution they meet at a Spike site. This
exapnds the table by including every inverse mutation (`S:XnY` → `S:YnX`, 
score `1/s`), and transitions chain (`X→Y` plus `Y→Z` gives `X→Z`, score `s₁·s₂`). 
Its only input is the PyR<sub>0</sub> `mutations.tsv` you downloaded.

```bash
python scripts/pyro_aug.py
```

Arguments:
| Flag | Config default | Meaning |
| --- | --- | --- |
| `--pyro-version` | `fitness.pyro_version` | Release directory under `<data_root>/<fitness.pyro_subdir>`; sets both paths below. |
| `--mutations` | `fitness.pyro_mutations_filename` | Input table (`mutations.tsv`). |
| `-o`, `--output` | `fitness.pyro_spike_filename` | Output pickle (`PyroSpike.pkl`). |
| `--summary` | off | Also print the per-site table (count and score range per Spike site). |

### Steps 5–9 — Notebooks

```bash
jupyter lab --no-browser --port=8888
```

Run the notebooks in `scripts/` in this order:
1. `create_from_chronumental_data` (step 5a) **or** `create_from_metadata` (5b)
   — 5b needs no `chronumental` run, but choosing it also rules out step 8; see
   step 0
2. `rank_analysis`                                            (step 6)
3. `new_mutations_cumulative`                                 (step 7)
4. `new_mutations_fitness`                                    (step 8)
5. `scout_selection`                                          (step 9)

`pyro_aug.py` is not in this list: it is step 4, a command-line script already
run by `scripts/run_mat_pipeline.sh`.

They read the same `config.yaml`, so the only per-run knobs are in
the `analysis:` block — `start_date`/`end_date` (the window the tree is trimmed
to, applied in step 5a/5b), `cutoff_date`, `eve_backbone` and `beta`.

To sweep several cutoff dates without editing the tracked config:
```bash
cp config.yaml config-2023.yaml     # set analysis.cutoff_date: '2023-06-01'
VACCINESCOUT_CONFIG=$PWD/config-2023.yaml jupyter lab
```

## <a name="layout"></a> Repository Layout

```text
config.yaml                  All paths and run parameters
envs/                        Conda environment specifications
scripts/
  config_loader.py           Reads config.yaml; run directly to audit paths
                             (critical inputs vs generated intermediates)
  Helpers.py                 Cohort splitting, mutation parsing, fitness scoring
  run_mat_pipeline.sh        Driver for steps 0-4 (chronumental + the four below)
  get_amino_acid_mutations_of_sequences.py   MAT -> Spike AA mutations
  get_lineage_fasta.py                       MAT -> lineage_spike.fasta
  lineage_spike_from_mat.py                  MAT -> FDA strain sequences
  pyro_aug.py                                PyR0 mutations.tsv -> Spike pickle
  min_dist.cpp / pairwise_dist.cpp           Hamming distance kernels
  CMakeLists.txt
  *.ipynb                    Analysis notebooks (see the table above)
```

## <a name="troubleshooting"></a> Troubleshooting

**`Could not find 'config.yaml'`** — run from inside the repository, or set
`$VACCINESCOUT_CONFIG` to its absolute path.

**`min_dist not found` / `pairwise_dist not found`** — the binaries were never
built, or `paths.bin_dir` is wrong. See [step 3](#build).

**`Could not find the 'pyrocov' package`** — clone `pyro-cov` and point
`paths.pyro_cov` at it.

**`chronumental: command not found`** — it is not part of
`envs/vaccinescout.yml`; install it separately.

**`--use_gpu` runs on CPU** — check with
`python -c "import jax; print(jax.devices())"`. `[CpuDevice(id=0)]` means you
have the CPU `jaxlib`; install the CUDA wheel as above. Note that chronumental
reports `Platform: cpu` for any command line without `--use_gpu`, including
`--help`, which is expected.

**`covfit_cli` not found, or CoVFit steps fail immediately** — the notebooks run
`./covfit_cli` with the working directory set to `paths.covfit_cli`. 

##  <a name="cite"></a> Citing VaccineScout
