#!/usr/bin/env bash
#
# VaccineScout steps 0-4: the command-line part of the pipeline, before the
# notebooks. Steps 0-3 read the mutation-annotated tree; step 4 prepares the
# PyR0 fitness table, which is independent of the tree but is needed from
# notebook step 6 onwards.
#
#   conda activate vaccinescout
#   ./scripts/run_mat_pipeline.sh
#
# Every path and parameter comes from config.yaml, so this takes no arguments.
# Override the config location with $VACCINESCOUT_CONFIG.
#
#   Step 0  chronumental                             -> generates files.chronumental_dates
#   Step 1  get_amino_acid_mutations_of_sequences.py -> generates files.node_spike_cumulative
#                                                       generates files.node_spike_branch
#                                                       generates files.parent_node
#   Step 2  get_lineage_fasta.py                     -> generates files.lineage_spike_fasta
#   Step 3  lineage_spike_from_mat.py                -> generates files.fda_recommendations
#   Step 4  pyro_aug.py                              -> generates the PyR0 Spike pickle
#                                                       <data_root>/<fitness.pyro_subdir>/
#                                                       <pyro_version>/<pyro_spike_filename>
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Pull the paths we need out of config.yaml via the same loader the scripts use.
# Captured first so a failure here reports the real error instead of cascading
# into "unbound variable" further down.
if ! CONFIG_VARS="$(python - <<'PY'
import sys, shlex
sys.path.insert(0, "scripts")
from config_loader import load_config
c = load_config()
out = {
    "TREE":        c.file("tree_newick"),
    "METADATA":    c.file("metadata"),
    "DATES_OUT":   c.file("chronumental_dates"),
    "REF_NODE":    c.get("chronumental", "reference_node", ""),
    "STEPS":       c.get("chronumental", "steps", 2000),
    "USE_GPU":     "1" if c.get("chronumental", "use_gpu", True) else "",
    "RUN_CHRON":   "1" if c.run_chronumental else "",
}
for k, v in out.items():
    print(f"{k}={shlex.quote(str(v))}")
PY
)"; then
  echo "" >&2
  echo "Could not read config.yaml with '$(command -v python || echo python)'." >&2
  echo "Activate the pipeline environment first:  conda activate vaccinescout" >&2
  exit 1
fi
eval "$CONFIG_VARS"

step() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

# --- Step 0: date every node on the tree ------------------------------------
step "Step 0/4  chronumental"
if [[ -z "$RUN_CHRON" ]]; then
  echo "skipped: chronumental.run is false in config.yaml."
  echo
  echo "  No chronumental inferred dates will be produced, so create_from_chronumental_data.ipynb"
  echo "  (step 5a) has nothing to read. Run create_from_metadata.ipynb (step 5b)"
  echo "  instead."
  echo
  echo "  Set chronumental.run: true to restore step 0."
else
  for f in "$TREE" "$METADATA"; do
    [[ -s "$f" ]] || { echo "missing input: $f" >&2; exit 1; }
  done
  # --dates_out is passed explicitly: left to its default, chronumental appends
  # ".tsv" to the --dates filename and you get a confusing "...metadata.tsv.tsv".
  cmd=(chronumental --tree "$TREE" --dates "$METADATA"
       --reference_node "$REF_NODE" --dates_out "$DATES_OUT"
       --only_use_full_dates --steps "$STEPS")
  [[ -n "$USE_GPU" ]] && cmd+=(--use_gpu)
  echo "+ ${cmd[*]}"
  "${cmd[@]}"
fi

# --- Steps 1-3: MAT -> Spike mutations, lineage FASTA, FDA strains ----------
step "Step 1/4  get_amino_acid_mutations_of_sequences.py"
python scripts/get_amino_acid_mutations_of_sequences.py

step "Step 2/4  get_lineage_fasta.py"
python scripts/get_lineage_fasta.py

step "Step 3/4  lineage_spike_from_mat.py"
python scripts/lineage_spike_from_mat.py

# --- Step 4: expand the PyR0 mutation table into the Spike score pickle ------
step "Step 4/4  pyro_aug.py"
python scripts/pyro_aug.py

if [[ -n "$RUN_CHRON" ]]; then
  printf '\n\033[1mDone.\033[0m Continue with the notebooks, starting at step 5a\n'
  printf '(create_from_chronumental_data.ipynb).\n'
else
  printf '\n\033[1mDone.\033[0m chronumental was skipped, so start the notebooks at\n'
  printf 'step 5b (create_from_metadata.ipynb), not 5a.\n'
fi
