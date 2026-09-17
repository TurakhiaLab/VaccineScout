<div align="center">

# VaccineScout: Computational Framework for Determining SARS-CoV-2 Vaccine Composition


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
  - [1. Clone repositories and install the dependencies](#clone)
  - [2. Create the environment](#envs)
  - [3. Build the C++ helpers](#build)
  - [4. Register the Jupyter kernel](#kernel)
- [Quick Start: Running VaccineScout using MAT](#quickstart)
- [Configuration](#config)
- [Running the Pipeline](#pipeline)
  - [Steps 0–4. MAT processing and PyR<sub>0</sub> table generation](#steps-0-4)
  - [Steps 5–9. Jupyter notebooks generating Figs. 2, 3, 4B, and 4C](#steps-5-9)
  - [Step 10. VaccineScout results across cutoff years (Supplementary Figs. 2, 3, and 6)](#step-10)
  - [Step 11. Lineage proportions across years (Fig. 4A)](#step-11)
- [Repository Layout](#layout)
- [Troubleshooting](#troubleshooting)
- [Citing VaccineScout](#cite)

<br>

## <a name="intro"></a> Introduction

VaccineScout is a computational framework that uses genomic surveillance data
and fitness estimates to project the future prevalence of circulating SARS-CoV-2
variants. It selects varying numbers of medoids (circulating variants) that 
minimize the spike sequence distance to these forecasted variants and generates a distance-versus-valency curve to select the optimal vaccine valency.

<div align="center">
   <img src="docs/images/VaccineScout_overview.svg" style="display: block; margin-left: auto; margin-right: auto; width: 100%;">
  <div> <b>Figure 1: Overview of VaccineScout.</b> </div>
</div>

## <a name="install"></a> Installation

### <a name="clone"></a> 1. Clone repositories and install the dependencies
Clone the VaccineScout repository, then set up PyR0 and CoVFit.

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

⚠️ You can unzip CoVFit elsewhere; if you do, point `paths.covfit_cli` in 
`config.yaml` at that directory instead.

### <a name="envs"></a> 2. Create the environment

```bash
conda env create -f envs/vaccinescout.yml
```

[Chronumental](https://github.com/theosanderson/chronumental) infers a date for
every node on the tree. You need it to regenerate `files.chronumental_dates`. But
you can skip this step if you want to infer haplotype dates from their names.

```bash
conda activate vaccinescout
conda install -y -c conda-forge cudatoolkit=11.8.0 "cudnn<9.0"
conda install -y -c bioconda chronumental
conda remove --force -y jax jaxlib

pip install "jax==0.4.26" "jaxlib==0.4.26+cuda12.cudnn89" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### <a name="build"></a> 3. Build the C++ helpers

`min_dist` and `pairwise_dist` compute substitution distances over the sequences
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

## <a name="quickstart"></a> Quick Start: Running VaccineScout using MAT

After installation, you can use VaccineScout to select vaccine variants 
for any cutoff date by running:

```bash
# Provide full path except the output_file name
python scripts/run_vaccinescout_from_mat.py --cutoff-date 2022-06-01 --mat public-latest.all.masked.pb.gz --reference NC_045512v2.fa --output_dir results --output_file vaccinescout_medoids.csv
```

⚠️ Public SARS-CoV-2 MATs can be downloaded from [here](https://hgdownload.gi.ucsc.edu/goldenPath/wuhCor1/UShER_SARS-CoV-2/). Chronumental is not required when using these MATs, as haplotype dates are extracted directly from their names.

By default each leaf is dated from the trailing date field of its name. A full
`YYYY-MM-DD` is used as given, and a `YYYY-MM` is taken as the first of that
month. Only a bare year (`YYYY`) or an unparseable value is dropped. Two alternatives:

```bash
# Provide an external metadata file, which can be downloaded alongside the MAT
python scripts/run_vaccinescout_from_mat.py --cutoff-date 2022-06-01 \
  --metadata public-latest.metadata.tsv.gz --mat public-latest.all.masked.pb.gz --output_dir results --output_file vaccinescout_medoids.csv --reference NC_045512v2.fa

# Dates generated from a chronumental run (step 0)
python scripts/run_vaccinescout_from_mat.py --cutoff-date 2022-06-01 \
  --dates chronumental_dates_public-latest.metadata.tsv --mat public-latest.all.masked.pb.gz --output_dir results --output_file vaccinescout_medoids.csv --reference NC_045512v2.fa
```

## <a name="config"></a> Configuration

Every path and run parameter is defined in **`config.yaml`** at the 
repository root and read through `scripts/config_loader.py`. Print 
the current configuration and check which inputs are present:

```bash
python scripts/config_loader.py
```

This lists only what the pipeline reads as input, so anything
reported MISSING is a hard stop. It comes in two groups:

* **Input files** — the MAT tree (`.pb`), the Newick tree (`.nwk`) and 
  the corresponding metadata TSV, the reference genome (`NC_045512v2.fa`),
  the PyR<sub>0</sub> (`mutations.tsv`), the EVEscape (`eve_backbone`), 
  and a Pfizer genome FASTA per cutoff (`pfizer_fasta_by_cutoff`).
* **Directories and executables** — `data_root`, `pyro_cov`, `covfit_cli`, 
  `pfizer_dir`, and `min_dist` / `pairwise_dist` binaries.

## <a name="pipeline"></a> Running the Pipeline

Run the twelve steps in this order. All of them run in the single
`vaccinescout` environment.

| # | Step | Figure |
| --- | --- | --- |
| 0 | [`chronumental`](#step-0) | — |
| 1 | [`get_amino_acid_mutations_of_sequences.py`](#step-1) | — |
| 2 | [`get_lineage_fasta.py`](#step-2) | — |
| 3 | [`lineage_spike_from_mat.py`](#step-3) | — |
| 4 | [`pyro_aug.py`](#step-4) | — |
| 5a | [`create_from_chronumental_data.ipynb`](#steps-5-9) *(recommended)* | — |
| 5b | [`create_from_metadata.ipynb`](#steps-5-9) *(alternative to 5a)* | — |
| 6 | [`rank_analysis.ipynb`](#steps-5-9) | Fig. 2 |
| 7 | [`vaccinescout_selection.ipynb`](#steps-5-9) | Fig. 3 |
| 8 | [`new_mutations_cumulative.ipynb`](#steps-5-9) | Fig. 4B |
| 9 | [`new_mutations_fitness.ipynb`](#steps-5-9) | Fig. 4C |
| 10 | [`vaccinescout_results_across_years.py`](#step-10) | Supplementary Figs. 2, 3, and 6 |
| 11 | [`plot_lineage_proportions.py`](#step-11) | Fig. 4A |

⚠️ The results in the manuscript were obtained using a SARS-CoV-2 MAT that included sequences from GISAID, which are subject to data-sharing restrictions and therefore cannot be distributed by us. However, similar results can be obtained using publicly available SARS-CoV-2 MATs, which can be downloaded from [here](https://hgdownload.gi.ucsc.edu/goldenPath/wuhCor1/UShER_SARS-CoV-2/).

### <a name="steps-0-4"></a> Steps 0–4 — MAT processing and PyR<sub>0</sub> table generation

All five steps run from a single script that reads `config.yaml`:

```bash
conda activate vaccinescout
# Runs steps 0-4
./scripts/run_mat_pipeline.sh            
```

<a name="step-0"></a>

**Step 0** — Date every node on the tree with
[Chronumental](https://github.com/theosanderson/chronumental). 

Step 5a needs this result, while its alternate, step 5b does not.

```bash
# Running standalone. 
chronumental --tree <tree_newick> --dates <metadata> \
  --reference_node 'CHN/Wuhan_IME-WH01/2019|MT291826.1|2019-12-30' \
  --dates_out <chronumental_dates> \
  --only_use_full_dates --steps 2000 --use_gpu
```

⚠️ You can skip this step if you want to derive haplotype dates from their names
instead. Set `chronumental.run: false` in `config.yaml` and run **step 5b** 
(`create_from_metadata.ipynb`) rather than step 5a. But it will prevent you 
from running `Step 8 - new_mutations_fitness.ipynb`, which requires dates for 
internal nodes.

<a name="step-1"></a>

**Step 1** — Generating spike amino-acid mutations

Writes the cumulative and branch-specific spike mutations for each node, 
along with the parent–child relationships. It also writes the reference 
spike protein FASTA.

```bash
# Running standalone
python scripts/get_amino_acid_mutations_of_sequences.py
```

<a name="step-2"></a>

**Step 2** — Spike sequence per lineage

Identifies each lineage root, merges lineages with identical sets of 
spike mutations, and writes each merged group as a single FASTA 
sequence. Headers for merged lineages join the lineage names with `|`.

```bash
# Running standalone
python scripts/get_lineage_fasta.py
```

<a name="step-3"></a>

**Step 3** — Spike sequences for FDA-recommended vaccine variants

For each FDA-recommended vaccine variant in the given list, identifies
the corresponding lineage root and extracts its spike sequence.

```bash
# Running standalone
python scripts/lineage_spike_from_mat.py
```

<a name="step-4"></a>

**Step 4** — PyR<sub>0</sub> Spike score table update

The PyR<sub>0</sub> table contains scores for a limited set of mutations,
but the notebooks require scores for many additional substitutions. This 
expands the table by including inverse mutations 
(`S:XnY` → `S:YnX`, score `1/s`), and chained transitions 
(`X→Y` plus `Y→Z` gives `X→Z`, score `s₁·s₂`). 

```bash
# Running standalone
python scripts/pyro_aug.py
```

### <a name="steps-5-9"></a> Steps 5–9 — Jupyter notebooks generating Figs. 2, 3, 4B, and 4C

```bash
jupyter lab --no-browser --port=8888
```

Run the notebooks in `scripts/` in this order:
1. `create_from_chronumental_data.ipynb` (step 5a) or `create_from_metadata.ipynb` (5b)
   — 5b needs no `chronumental` run, but choosing it also rules out step 8
2. `rank_analysis.ipynb (Fig. 2)`                                             (step 6)
3. `vaccinescout_selection.ipynb (Fig. 3)`                                    (step 7)
4. `new_mutations_cumulative.ipynb (Fig. 4B)`                                 (step 8)
5. `new_mutations_fitness.ipynb (Fig. 4C)`                                    (step 9)

### <a name="step-10"></a> Step 10 — VaccineScout results across cutoff years (Supplementary Figs. 2, 3, and 6)

This step runs VaccineScout (step 7) across different cutoff dates and beta values.
For each cutoff date–beta pair, it selects K = 1-8 medoids and computes their
corresponding lineages, CoVFit fitness estimates, spike sequence distances to 
sequences that appeared 0-6 and 6-12 months later, and spike sequence distances to
the forecatsed prevalence of circulating variants. It compares the selected 
VaccineScout medoids on these metrics with the FDA-recommended variants, Pfizer 
vaccine variants, and an `oracle` medoid set selected using the true future variant 
prevalence.

```bash
python scripts/vaccinescout_results_across_years.py
```

Generated outputs:
1. `vaccinescout_medoid_report.xlsx` — one sheet per cutoff date, one column
   block per beta.
2. `vaccinescout_medoid_report_plots/distance_to_future_0_6_vs_k.svg` —
   VaccineScout vs oracle vs FDA/Pfizer distance to the sequences that appeared
   0-6 months later, for K = 1-8.
3. `vaccinescout_medoid_report_plots/distance_to_future_6_12_vs_k.svg` — the
   same, for the 6-12 month horizon.
4. `vaccinescout_medoid_report_plots/distance_to_current_vs_k.svg` — fitness 
   weighted distance to the variants circulating at the cutoff, for K = 1-8.
5. `vaccinescout_medoid_report_plots/distance_to_future_vs_beta_<cutoff>.svg` —
   distance to future sequences against beta, one file per cutoff date, for
   K = 1-4.

A full run scores every cutoff year with CoVFit and rebuilds the pairwise distance
matrices, which is slow. To re-run on existing intermediates, or to narrow the
sweep:

```bash
# reuse existing CoVFit predictions and pairwise_dist
python scripts/vaccinescout_results_across_years.py --skip-covfit --skip-pairwise

# one cutoff, one beta
python scripts/vaccinescout_results_across_years.py --cutoff-dates 2024-06-01 --betas 60

# plots somewhere else
python scripts/vaccinescout_results_across_years.py --plot-dir figures/
```

### <a name="step-11"></a> Step 11 — Lineage proportions across years (Fig. 4A)

This plots the monthly proportion of circulating Pango lineages as a stacked 
bar chart. Lineages with identical spike protein sequences are combined, such 
as `BA.4/BA.5`. The plot is written to 
`<paths.data_root>/lineage_proportions_across_years.svg`.

```bash
python scripts/plot_lineage_proportions.py
```

## <a name="layout"></a> Repository Layout

```text
config.yaml                  All paths and run parameters
envs/                        Conda environment specifications
scripts/
  config_loader.py                           Reads config.yaml; run directly to audit paths
  Helpers.py                                 Cohort splitting, mutation parsing, fitness scoring
  run_mat_pipeline.sh                        Driver for steps 0-4 (chronumental + the four below)
  run_vaccinescout_from_mat.py               medoid CSV
  get_amino_acid_mutations_of_sequences.py   Spike amino acid mutations
  get_lineage_fasta.py                       lineage spike sequences
  lineage_spike_from_mat.py                  FDA strain sequences
  pyro_aug.py                                Adding spike mutations scored by PyR0
  vaccinescout_results_across_years.py       vaccinescout_medoid_report.xlsx and plots
  plot_lineage_proportions.py                lineage_proportions_across_years.svg
  min_dist.cpp / pairwise_dist.cpp           Substitution distance kernels
  CMakeLists.txt
  *.ipynb                    Analysis notebooks (steps 5-9)
```

## <a name="troubleshooting"></a> Troubleshooting

1. `Could not find 'config.yaml'` — run from inside the repository, or set
`$VACCINESCOUT_CONFIG` to its absolute path.

2. `min_dist not found` / `pairwise_dist not found` — the binaries were never
built, or `paths.bin_dir` is wrong. See [step 3](#build).

3. `Could not find the 'pyrocov' package` — clone `pyro-cov` from [step 1](#clone) and point
`paths.pyro_cov` at it.

4. `chronumental: command not found` — install it by following [step 2](#envs).

5. `covfit_cli` not found, or CoVFit steps fail immediately — the notebooks run
`./covfit_cli` with the working directory set to `paths.covfit_cli`. 

##  <a name="cite"></a> Citing VaccineScout
