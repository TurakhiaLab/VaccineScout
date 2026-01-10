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
- [Introduction](#intro) ([Wiki])
- [Installation](#install)
- [Quick Start](#example)
- [User Guide](#guide)
  - [Organizing Data](#data)
  - [Scout Arguments](#arguments)
  - [Run Command](#run)
  - [Getting MATs](#mat)
- [Contributions](#contribution)
- [Citing Scout](#cite)

<br>


## <a name="intro"></a> Introduction
To be Added
## <a name="install"></a> Installation
To be Added

##  <a name="example"></a> Quick Start
The following steps will reproduce the results and figures described in the manuscript.

### <a name="reproduce_plots"></a> Strain Scouting for Cutoff Date: 7/1/2022
**Step 1:** Download the appropriate MAT
```bash
<command pending>
```
Download a MAT with at least 1 year of data beyond the cutoff date.

**Step 2:**  Setup Chronumental
```bash
<command pending>
```

**Step 3:**  Process Raw MAT
```bash
<file missing>
```

**Step 4:**  Obtain Date annotation for MAT
```bash
<command pending>
```

**Step 5:**  Obtain processed DataFrame
Run the notebook - create_from_chronumental_data.ipynb and obtain final_df and other helper files

**Step 6:**  For Overlap Trend Analysis
Run the notebook - spike_overlap_analysis.ipynb and reproduce the figure below

**Step 7:**  For Compartivie Ranking Analysis
Run the notebook - rank_analysis.ipynb and reproduce the figure below

**Step 8:**  For Scout Selection 
Run the notebooks in the following order
```bash
pairwise_dist.ipynb
scout_selection.ipynb
```
and reproduce the figures below

## <a name="guide"></a> User Guide
### <a name="data"></a> Organizing Data
Visualization of Scout's workflow directories
```text

```

### <a name="arguments"></a> Scout Arguments


###  <a name="mat"></a> Getting Mutation-Annotated Trees
Mutation-annotated trees (MAT) for different pathogens are maintained by the UShER team, which can be found [here](https://dev.usher.bio).

##  <a name="contribution"></a> Contributions


##  <a name="cite"></a> Citing VaccineScout
