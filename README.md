<div align="center">

# VaccineScout: A Computational Framework for Guiding the Composition and Valency of SARS-CoV-2 Vaccine Updates


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

VaccineScout is a computational tool that addresses two key questions 
in SARS-CoV-2 vaccine selection: **which** variants to include, and 
**how many**. VaccineScout requires SARS-CoV-2 genomic surveillance 
data in the form of **mutation-annotated trees** [MATs](https://github.com/yatisht/usher) 
— daily-updated phylogeny containing all publicly available SARS-CoV-2 
sequences, together with **variant fitness estimates**. 

VaccineScout reads a MAT and extracts the spike amino acid sequences of 
circulating SARS-CoV-2 variants, as the vaccines target the spike protein. 
It then combines current variant prevalence with fitness estimates from 
an existing model, [CoVFit](https://github.com/TheSatoLab/CoVFit) by default, 
to project the future prevalence of circulating variants.

Using these projected prevalences, VaccineScout selects *k* medoids as 
candidate vaccine compositions. The medoids are chosen to minimize the 
prevalence-weighted average spike sequence distance between projected 
variants and their nearest selected medoid. VaccineScout evaluates this 
objective across different values of *k*. Increasing *k* increases 
vaccine complexity and cost while reducing the distance to projected 
variants, with diminishing returns as additional variants are included. 
Plotting this trade-off as distance versus *k* produces a knee point 
that provides a data-driven basis for selecting vaccine valency.


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
conda install -y -c bioconda chronumental
```

⚠️ You can append `--solver=libmamba` to `conda install` command to speed up dependency resolution and installation.


If your system does not have a GPU (checked with `nvidia-smi`), set `use_gpu: false` in both the  `covfit` and `chronumental` section of `config.yaml`. Otherwise, if `nvidia-smi` runs successfully and reports an NVIDIA driver version of 525.60 or later, run the following commands.

```bash
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

After installation, VaccineScout can give vaccine recommendations for any cutoff 
date directly from the SARS-CoV-2 MAT. The example below uses the Sept 1, 2026
public tree.

**1. Download the MAT** 
```bash
wget https://hgdownload.gi.ucsc.edu/goldenPath/wuhCor1/UShER_SARS-CoV-2/2026/09/01/public-2026-09-01.all.masked.pb.gz
```

**2. Run VaccineScout**:
```bash
python scripts/run_vaccinescout_from_mat.py --cutoff-date 2026-06-01 \
  --mat public-2026-09-01.all.masked.pb.gz \
  --reference data/NC_045512v2.fa \
  --output_dir results --output_file vaccinescout_2026-06-01.csv
```

**3. Check outputs** 
Under `results/`
| Output | Contents |
| --- | --- |
| `vaccinescout_2026-06-01.csv` | one row per selected medoid: medoid ID, CoVFit fitness, average minimum spike-sequence distance between the forecasted prevalence of circulating variants and the selected candidates, and corresponding PANGO lineage(s) |
| `medoids/medoids_k1.fasta` … `medoids/medoids_k8.fasta` | the selected medoid spike sequences, with one FASTA per selected K |
| `medoids/medoids_mutations.csv` | mutations of the selected medoids |

Expected output in `vaccinescout_2026-06-01.csv`
```text
k,medoid,covfit_fitness,distance_current,lineage
1,S_776c44ac2725,,9.558510638297875,XFG.1.1|XFG.1.1.1|XFG.1.1.10|XFG.1.1.4|XFG.1.1.6|XFG.1.1.9|XFG.1.1.5|XFG.1.1.8|XFG.1.1.11
-----
2,S_776c44ac2725,,3.9521276595744688,XFG.1.1|XFG.1.1.1|XFG.1.1.10|XFG.1.1.4|XFG.1.1.6|XFG.1.1.9|XFG.1.1.5|XFG.1.1.8|XFG.1.1.11
2,S_cb542f5cf466,,3.9521276595744688,RE.2.5|SK.1|RE.2.2.3|RS.1.3|RE.1.1.1|RE.2|RE.2.1.5|RE.2.2.2|RE.2.3|SQ.2
-----
3,S_776c44ac2725,,2.797872340425532,XFG.1.1|XFG.1.1.1|XFG.1.1.10|XFG.1.1.4|XFG.1.1.6|XFG.1.1.9|XFG.1.1.5|XFG.1.1.8|XFG.1.1.11
3,S_cb542f5cf466,,2.797872340425532,RE.2.5|SK.1|RE.2.2.3|RS.1.3|RE.1.1.1|RE.2|RE.2.1.5|RE.2.2.2|RE.2.3|SQ.2
3,S_8cafe31ebd39,,2.797872340425532,PQ.2.8.1|RC.3|RC.5|PQ.2.7.2
-----
```
⚠️ Public SARS-CoV-2 MATs and their metadata can be downloaded from
 [here](https://hgdownload.gi.ucsc.edu/goldenPath/wuhCor1/UShER_SARS-CoV-2/). 
  Chronumental is not required to run VaccineScout, but is recommended because
  it can assign dates to haplotypes that lack complete dates in their names. 
  Without an external dates file, VaccineScout extracts dates from each 
  haplotype name by searching for the `YYYY-MM-DD` pattern. It also accepts 
  the `YYYY-MM` format and assigns the first day of the corresponding month. 
  Haplotypes with only a bare year (`YYYY`) or an otherwise unparseable date 
  are excluded from the analysis. These cases can be addressed using either 
  of the following methods:

```bash
# Dates generated from a chronumental run (RECOMMENDED)
python scripts/run_vaccinescout_from_mat.py --cutoff-date 2022-06-01 \
  --dates <path_to>/chronumental_dates_public-latest.metadata.tsv
  --mat <path_to>/public-latest.all.masked.pb.gz --reference <path_to>/NC_045512v2.fa \
  --output_dir <path_to>/<output_folder> --output_file <file_name.csv>

# Provide an external metadata file containing dates, which can be downloaded alongside the MAT
python scripts/run_vaccinescout_from_mat.py --cutoff-date 2022-06-01 \
  --metadata <path_to>/public-latest.metadata.tsv.gz \
  --mat <path_to>/public-latest.all.masked.pb.gz --reference <path_to>/NC_045512v2.fa \
  --output_dir <path_to>/<output_folder> --output_file <file_name.csv>
```

## <a name="config"></a> Configuration

Every path and run parameter is defined in **`config.yaml`** at the 
repository root and read through `scripts/config_loader.py`. Print 
the current configuration and check which inputs are present:

```bash
python scripts/config_loader.py
```

This lists only what the pipeline reads as input, so anything
reported MISSING is a hard stop. There are two groups:

* **Input files** — MAT (`mat`), Newick tree (`tree_newick`) and 
  corresponding metadata (`metadata`), reference genome (`reference`),
  PyR<sub>0</sub> (`pyro_mutations_filename`), EVEscape (`eve_backbone`), 
  and Pfizer genomes FASTA per cutoff (`pfizer_fasta_by_cutoff`).
* **Directories and executables** — `data_root`, `pyro_cov`, `covfit_cli`, 
  `pfizer_dir`, and `min_dist` and `pairwise_dist` binaries.

⚠️ The publicly available SARS-CoV-2 MAT, Newick tree, and associated 
metadata can be downloaded from 
[here](https://hgdownload.gi.ucsc.edu/goldenPath/wuhCor1/UShER_SARS-CoV-2). 
Download the files to the `data` directory and update the corresponding 
`mat`, `tree_newick` and `metadata` variables in `config.yaml`.

## <a name="pipeline"></a> Running the Pipeline

Run the twelve steps in the order shown below. All steps should be run 
within the same `vaccinescout` environment.

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
| 7 | [`vaccinescout_selection.ipynb`](#steps-5-9) | Fig. 3, Supplementary Fig. 1 (Update `cutoff_date`) |
| 8 | [`new_mutations_cumulative.ipynb`](#steps-5-9) | Fig. 4B, Supplementary Fig. 4 (Update `cutoff_date`) |
| 9 | [`new_mutations_fitness.ipynb`](#steps-5-9) | Fig. 4C, Supplementary Figs. 5 and 6 (Update `cutoff_date`) |
| 10 | [`vaccinescout_results_across_years.py`](#step-10) | Supplementary Figs. 2, 3, and 7 |
| 11 | [`plot_lineage_proportions.py`](#step-11) | Fig. 4A |

⚠️ The results in the manuscript were obtained using a SARS-CoV-2 MAT 
that included sequences from GISAID, which are subject to data-sharing 
restrictions and therefore cannot be distributed by us. However, 
similar results can be obtained using publicly available SARS-CoV-2 
MATs, which can be downloaded from 
[here](https://hgdownload.gi.ucsc.edu/goldenPath/wuhCor1/UShER_SARS-CoV-2/).

### <a name="steps-0-4"></a> Steps 0–4 — MAT processing and PyR<sub>0</sub> table generation

All five steps run from a single script that reads `config.yaml`.

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
sweep for getting faster results:

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

| Path | Purpose | Step |
| --- | --- | --- |
| [`config.yaml`](config.yaml) | All paths and run parameters | — |
| [`envs/`](envs/) | Conda environment specifications | — |
| [`scripts/config_loader.py`](scripts/config_loader.py) | Reads `config.yaml`; run directly to audit paths | — |
| [`scripts/Helpers.py`](scripts/Helpers.py) | Cohort splitting, mutation parsing, fitness scoring | — |
| [`scripts/run_mat_pipeline.sh`](scripts/run_mat_pipeline.sh) | Driver for the four scripts below, plus `chronumental` | [0–4](#steps-0-4) |
| [`scripts/get_amino_acid_mutations_of_sequences.py`](scripts/get_amino_acid_mutations_of_sequences.py) | Spike amino acid mutations | [1](#step-1) |
| [`scripts/get_lineage_fasta.py`](scripts/get_lineage_fasta.py) | Lineage spike sequences | [2](#step-2) |
| [`scripts/lineage_spike_from_mat.py`](scripts/lineage_spike_from_mat.py) | FDA strain sequences | [3](#step-3) |
| [`scripts/pyro_aug.py`](scripts/pyro_aug.py) | Adding spike mutations scored by PyR<sub>0</sub> | [4](#step-4) |
| [`scripts/*.ipynb`](scripts/) | Analysis notebooks | [5–9](#steps-5-9) |
| [`scripts/vaccinescout_results_across_years.py`](scripts/vaccinescout_results_across_years.py) | `vaccinescout_medoid_report.xlsx` and plots | [10](#step-10) |
| [`scripts/plot_lineage_proportions.py`](scripts/plot_lineage_proportions.py) | `lineage_proportions_across_years.svg` | [11](#step-11) |
| [`scripts/run_vaccinescout_from_mat.py`](scripts/run_vaccinescout_from_mat.py) | Medoid CSV and per-K medoid FASTAs, straight from a MAT | [Quick Start](#quickstart) |
| [`scripts/min_dist.cpp`](scripts/min_dist.cpp) / [`pairwise_dist.cpp`](scripts/pairwise_dist.cpp) | Substitution distance kernels | — |
| [`scripts/CMakeLists.txt`](scripts/CMakeLists.txt) | Build definition for the two kernels above | — |

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
