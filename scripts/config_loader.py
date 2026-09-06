#!/usr/bin/env python3
"""
Central configuration access for VaccineScout.

Every path and run parameter the pipeline needs lives in ``config.yaml`` at the
repository root. This module finds that file, resolves its relative paths
against the repo root, and exposes them through a small typed-ish accessor so
that no script or notebook has to hardcode a location.

Typical use from a notebook in ``scripts/``::

    from config_loader import load_config

    cfg = load_config()
    ROOT_PATH = cfg.data_root_str            # trailing-separator string
    final_df  = pd.read_parquet(cfg.file("final_df"))

Discovery order for the config file:

1. ``$VACCINESCOUT_CONFIG`` if set (must exist).
2. ``config.yaml`` in the current working directory or any ancestor.
3. ``config.yaml`` beside this module or any of its ancestors.

Relative paths in the ``paths:`` block resolve against the directory holding
the config file; ``~`` and ``$ENV_VARS`` are expanded everywhere.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence



def _import_yaml():
    """Import PyYAML lazily.

    Deferring this keeps ``import config_loader`` working in environments that
    lack PyYAML (the bioconda ``bte`` env, for instance), so scripts driven
    entirely by command-line flags still run. The error surfaces only when a
    configured value is actually needed.
    """
    try:
        import yaml  # noqa: PLC0415 - intentionally deferred
    except ModuleNotFoundError as exc:  # pragma: no cover - environment problem
        raise ModuleNotFoundError(
            "PyYAML is required to read config.yaml. Install it with:\n"
            "    conda install -c conda-forge pyyaml"
        ) from exc
    return yaml

CONFIG_FILENAME = "config.yaml"
ENV_VAR = "VACCINESCOUT_CONFIG"

__all__ = [
    "Config",
    "load_config",
    "load_config_optional",
    "find_config_file",
    "CONFIG_FILENAME",
    "ENV_VAR",
]


def _expand(value: str) -> str:
    """Expand ``~`` and environment variables in a path-like string."""
    return os.path.expanduser(os.path.expandvars(str(value)))


def find_config_file(start: Optional[os.PathLike] = None) -> Path:
    """
    Locate ``config.yaml``.

    Raises
    ------
    FileNotFoundError
        If no config file can be found, listing the directories searched.
    """
    override = os.environ.get(ENV_VAR)
    if override:
        path = Path(_expand(override)).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"${ENV_VAR} points at '{path}', which does not exist."
            )
        return path

    roots: List[Path] = []
    if start is not None:
        roots.append(Path(start))
    roots.append(Path.cwd())
    roots.append(Path(__file__).resolve().parent)

    searched: List[Path] = []
    for root in roots:
        root = root.resolve()
        for directory in (root, *root.parents):
            if directory in searched:
                continue
            searched.append(directory)
            candidate = directory / CONFIG_FILENAME
            if candidate.is_file():
                return candidate

    raise FileNotFoundError(
        f"Could not find '{CONFIG_FILENAME}'. Searched:\n  "
        + "\n  ".join(str(d) for d in searched)
        + f"\nSet ${ENV_VAR} to point at it explicitly."
    )


class Config:
    """Resolved view over ``config.yaml``."""

    def __init__(self, data: Dict[str, Any], config_path: Path):
        self._data = data or {}
        self.config_path = Path(config_path).resolve()
        self.repo_root = self.config_path.parent

    # -- raw section access -------------------------------------------------
    @property
    def raw(self) -> Dict[str, Any]:
        return self._data

    def section(self, name: str) -> Dict[str, Any]:
        value = self._data.get(name) or {}
        if not isinstance(value, dict):
            raise TypeError(f"Config section '{name}' is not a mapping.")
        return value

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Fetch ``config[section][key]``, falling back to ``default``."""
        return self.section(section).get(key, default)

    # -- path helpers -------------------------------------------------------
    def _resolve(self, value: str) -> Path:
        """Absolute paths as-is; relative ones against the repo root."""
        path = Path(_expand(value))
        return path if path.is_absolute() else (self.repo_root / path)

    def path(self, key: str) -> Path:
        """A directory from the ``paths:`` block."""
        paths = self.section("paths")
        if key not in paths:
            raise KeyError(
                f"paths.{key} is not defined in {self.config_path}. "
                f"Available: {sorted(paths)}"
            )
        return self._resolve(paths[key])

    @property
    def data_root(self) -> Path:
        return self.path("data_root")

    @property
    def data_root_str(self) -> str:
        """``data_root`` as a string with a trailing separator.

        Lets existing ``ROOT_PATH + "file.parquet"`` concatenations keep working.
        """
        return os.path.join(str(self.data_root), "")

    @property
    def covfit_dir(self) -> Path:
        return self.path("covfit_cli")

    @property
    def pyro_cov_dir(self) -> Path:
        return self.path("pyro_cov")

    @property
    def pfizer_dir(self) -> Path:
        return self.path("pfizer_dir")

    def file(self, key: str) -> Path:
        """A data file from the ``files:`` block, resolved under ``data_root``."""
        files = self.section("files")
        if key not in files:
            raise KeyError(
                f"files.{key} is not defined in {self.config_path}. "
                f"Available: {sorted(files)}"
            )
        value = Path(_expand(files[key]))
        return value if value.is_absolute() else (self.data_root / value)

    def binary(self, name: str) -> Path:
        """Path to a compiled helper (``min_dist`` / ``pairwise_dist``)."""
        candidate = self.path("bin_dir") / name
        if not candidate.is_file():
            raise FileNotFoundError(
                f"{name} not found at {candidate}. Build it from the repo root:\n"
                "    cmake -S scripts -B scripts/build -DCMAKE_BUILD_TYPE=Release\n"
                "    cmake --build scripts/build -j"
            )
        return candidate

    def cutoff_dir(self, cutoff_date: str, create: bool = True) -> Path:
        """Per-cutoff output directory ``<data_root>/<cutoff_date>``."""
        directory = self.data_root / str(cutoff_date)
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        return directory

    # -- fitness inputs -----------------------------------------------------
    @property
    def pyro_version(self) -> str:
        return self.get("fitness", "pyro_version")

    def pyro_spike_pickle(self, version: Optional[str] = None) -> Path:
        """``<data_root>/<pyro_subdir>/<version>/PyroSpike.pkl``."""
        version = version or self.pyro_version
        return (
            self.data_root
            / self.get("fitness", "pyro_subdir", "pyro_ver")
            / str(version)
            / self.get("fitness", "pyro_spike_filename", "PyroSpike.pkl")
        )

    def pyro_mutations_tsv(self, version: Optional[str] = None) -> Path:
        """Raw PyR0 mutation table ``<data_root>/<pyro_subdir>/<version>/mutations.tsv``."""
        version = version or self.pyro_version
        return (
            self.data_root
            / self.get("fitness", "pyro_subdir", "pyro_ver")
            / str(version)
            / self.get("fitness", "pyro_mutations_filename", "mutations.tsv")
        )

    def eve_scores(self, backbone: Optional[str] = None) -> Path:
        """``<data_root>/EVEscape/<backbone>``, from ``analysis.eve_backbone``."""
        backbone = backbone or self.get("analysis", "eve_backbone")
        if backbone is None:
            raise KeyError(
                "No EVEscape backbone configured. Set analysis.eve_backbone in "
                f"{self.config_path} (e.g. EVEscore_Omicron_XBB)."
            )
        return (
            self.data_root
            / self.get("fitness", "eve_subdir", "EVEscape")
            / str(backbone)
        )

    @property
    def fitness_column(self) -> str:
        return self.get("fitness", "fitness_column", "fitness_mean")

    # -- CoVFit -------------------------------------------------------------
    def covfit_outdir(self, cutoff_date: str, kind: str = "current",
                      create: bool = True) -> Path:
        """Output directory for a CoVFit run (``current``/``vaccine``/``pfizer``)."""
        key = f"{kind}_subdir"
        subdir = self.get("covfit", key)
        if subdir is None:
            raise KeyError(
                f"covfit.{key} is not defined in {self.config_path}. "
                "Expected one of kind='current'|'vaccine'|'pfizer'."
            )
        directory = self.cutoff_dir(cutoff_date, create=create) / subdir
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        return directory

    def covfit_predictions(self, outdir: os.PathLike,
                           fold: Optional[int] = None) -> Path:
        """The ``CoVFit_Predictions_fold_<n>.tsv`` inside a CoVFit output dir."""
        fold = self.get("covfit", "fold", 3) if fold is None else fold
        return Path(outdir) / f"CoVFit_Predictions_fold_{fold}.tsv"

    def covfit_command(self, fasta_in: os.PathLike, outdir: os.PathLike,
                       extra: Sequence[str] = ()) -> List[str]:
        """Build the ``covfit_cli`` argv, run from ``cfg.covfit_dir``."""
        covfit = self.section("covfit")
        cmd = [
            str(covfit.get("executable", "./covfit_cli")),
            "-i", str(fasta_in),
            "-o", str(outdir),
        ]
        if covfit.get("use_gpu", True):
            cmd.append("-g")
        cmd += ["-b", str(covfit.get("batch_size", 16))]
        cmd += ["-f", str(covfit.get("fold", 3))]
        cmd += list(extra)
        return cmd

    # -- analysis parameters ------------------------------------------------
    @property
    def analysis(self) -> Dict[str, Any]:
        return self.section("analysis")

    @property
    def run_chronumental(self) -> bool:
        """Whether pipeline step 0 runs. False means dating comes from tip names."""
        return bool(self.get("chronumental", "run", True))

    @property
    def start_date(self) -> str:
        """Earliest node date kept when the tree is trimmed."""
        return str(self.get("analysis", "start_date"))

    @property
    def end_date(self) -> str:
        """Latest node date kept when the tree is trimmed."""
        return str(self.get("analysis", "end_date"))

    @property
    def cutoff_date(self) -> str:
        return str(self.get("analysis", "cutoff_date"))

    @property
    def min_tip_count(self) -> int:
        """Drop sequences observed on fewer tips than this before ranking."""
        return int(self.get("analysis", "min_tip_count", 5))

    @property
    def beta(self):
        return self.get("analysis", "beta", 60)

    @property
    def disjoint_windows(self) -> List[str]:
        return list(self.get("analysis", "disjoint_windows", []))

    def split_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for ``Helpers.split_circulating_and_future``."""
        a = self.analysis
        return {
            "left_buffer_days": a.get("left_buffer_days", 4),
            "right_buffer_days": a.get("right_buffer_days", 4),
            "circulation_months": a.get("circulation_months"),
            "circulation_days": a.get("circulation_days", 30),
            "future_months": a.get("future_months", 12),
            "future_days": a.get("future_days"),
        }

    # -- vaccine reference strains -----------------------------------------
    def _by_cutoff(self, key: str, cutoff_date: str) -> Any:
        table = self.get("vaccines", key) or {}
        if cutoff_date not in table:
            raise KeyError(
                f"No vaccines.{key} entry for cutoff_date={cutoff_date!r}. "
                f"Known cutoffs: {sorted(table)}"
            )
        return table[cutoff_date]

    def fda_strain(self, cutoff_date: str) -> str:
        return self._by_cutoff("fda_strain_by_cutoff", cutoff_date)

    def pfizer_inn(self, cutoff_date: str) -> str:
        return self._by_cutoff("pfizer_inn_by_cutoff", cutoff_date)

    def pfizer_fasta(self, cutoff_date: str) -> Path:
        return self.pfizer_dir / self._by_cutoff(
            "pfizer_fasta_by_cutoff", cutoff_date
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Config {self.config_path} data_root={self.data_root}>"


_CACHE: Dict[Path, Config] = {}


def load_config(path: Optional[os.PathLike] = None, *,
                reload: bool = False) -> Config:
    """
    Load (and memoize) the VaccineScout configuration.

    Parameters
    ----------
    path : path-like, optional
        Explicit config file. Defaults to :func:`find_config_file`.
    reload : bool
        Re-read from disk instead of returning the cached instance. Useful in a
        long-running notebook after editing ``config.yaml``.
    """
    config_path = Path(path).resolve() if path is not None else find_config_file()

    if not reload and config_path in _CACHE:
        return _CACHE[config_path]

    yaml = _import_yaml()
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{config_path} must contain a YAML mapping at the top level.")

    cfg = Config(data, config_path)
    _CACHE[config_path] = cfg
    return cfg


def load_config_optional(path: Optional[os.PathLike] = None):
    """
    Best-effort configuration load that never raises.

    Returns ``(config, None)`` on success or ``(None, exception)`` when the
    config file or PyYAML is unavailable. The MAT command-line scripts use this
    so they keep working from explicit flags alone in a minimal environment
    (for example a bioconda ``bte`` env without PyYAML installed).
    """
    try:
        return load_config(path), None
    except Exception as exc:  # noqa: BLE001 - deliberately broad, best effort
        return None, exc


# ---------------------------------------------------------------------------
# Classification used by the audit below (`python scripts/config_loader.py`).
#
# CRITICAL entries are the pipeline's inputs: things you download, check out or
# build yourself. Nothing regenerates them, so a missing one is a hard stop.
#
# Everything else is an intermediate that some pipeline step writes. Missing
# only means that step has not run yet, which is the normal state of a fresh
# checkout -- hence the separate section, so a wall of MISSING lines does not
# bury the handful that actually need your attention.
# ---------------------------------------------------------------------------

#: ``paths:`` keys that must already exist.
CRITICAL_PATHS = ("data_root", "pyro_cov", "covfit_cli", "pfizer_dir", "bin_dir")

#: ``files:`` keys that are inputs rather than pipeline products.
CRITICAL_FILES = ("reference", "mat", "tree_newick", "metadata")

#: Which step writes each remaining ``files:`` entry.
PRODUCED_BY = {
    "chronumental_dates":         "step 0  chronumental",
    "ref_spike_genome":           "step 1  get_amino_acid_mutations_of_sequences.py",
    "node_spike_cumulative":      "step 1  get_amino_acid_mutations_of_sequences.py",
    "node_spike_branch":          "step 1  get_amino_acid_mutations_of_sequences.py",
    "parent_node":                "step 1  get_amino_acid_mutations_of_sequences.py",
    "lineage_spike_fasta":        "step 2  get_lineage_fasta.py",
    "fda_recommendations":        "step 3  lineage_spike_from_mat.py",
    "final_df":                   "step 5  create_from_chronumental_data.ipynb",
    "sequence_dict":              "step 5  create_from_chronumental_data.ipynb",
    "mutation_path_dict":         "step 5  create_from_chronumental_data.ipynb",
    "rank_current_fasta":         "step 6  rank_analysis.ipynb",
    "current_id_to_seq":          "step 9  scout_selection.ipynb",
    "current_seq_mutation_sites": "step 9  scout_selection.ipynb",
    "pairwise_dist":              "step 9  scout_selection.ipynb (pairwise_dist)",
    "vaccinescout_current_fasta": "step 9  scout_selection.ipynb",
    "fda_strains_fasta":          "step 9  scout_selection.ipynb",
}


def _audit_rows(cfg: "Config"):
    """``(critical, generated)`` rows of ``(label, note, path)`` for the audit."""
    critical = []
    for key in CRITICAL_PATHS:
        try:
            critical.append((key, "directory", cfg.path(key)))
        except Exception as exc:  # noqa: BLE001 - report, do not abort the audit
            critical.append((key, f"unresolved: {exc}", None))

    for key in CRITICAL_FILES:
        critical.append((key, "downloaded input", cfg.file(key)))

    for label, note, getter in (
        ("PyR0 mutations tsv",
         f"fitness.pyro_version={cfg.get('fitness', 'pyro_version')}",
         cfg.pyro_mutations_tsv),
        ("EVEscape scores", f"analysis.eve_backbone={cfg.get('analysis', 'eve_backbone')}",
         cfg.eve_scores),
    ):
        try:
            critical.append((label, note, getter()))
        except Exception as exc:  # noqa: BLE001
            critical.append((label, f"unresolved: {exc}", None))

    cutoff = cfg.cutoff_date
    try:
        critical.append((f"Pfizer FASTA", f"analysis.cutoff_date={cutoff}",
                         cfg.pfizer_fasta(cutoff)))
    except Exception as exc:  # noqa: BLE001
        critical.append(("Pfizer FASTA", f"unresolved: {exc}", None))

    # binary() raises when a helper is missing, which is exactly what an audit
    # must not do -- resolve the paths by hand instead.
    executable = str(cfg.get("covfit", "executable", "./covfit_cli"))
    critical.append(("covfit_cli", "covfit.executable",
                     cfg.path("covfit_cli") / Path(executable).name))
    for name in ("min_dist", "pairwise_dist"):
        critical.append((name, "built by cmake", cfg.path("bin_dir") / name))

    generated = []
    try:
        generated.append(("PyR0 spike pickle", "step 4  pyro_aug.py",
                          cfg.pyro_spike_pickle()))
    except Exception as exc:  # noqa: BLE001
        generated.append(("PyR0 spike pickle", f"unresolved: {exc}", None))

    for key in cfg.section("files"):
        if key in CRITICAL_FILES:
            continue
        generated.append((key, PRODUCED_BY.get(key, "not written by any known step"),
                          cfg.file(key)))
    # Pipeline order reads better than alphabetical; the step labels sort into it,
    # and anything unrecognised falls to the end.
    generated.sort(key=lambda row: (row[1].startswith("step") is False, row[1], row[0]))
    return critical, generated


def _shorten(cfg: "Config", resolved: Optional[Path]) -> str:
    """Show a path relative to the data root or repo root when it sits inside one."""
    if resolved is None:
        return ""
    for base, prefix in ((cfg.data_root, "<data_root>"), (cfg.repo_root, "<repo>")):
        if resolved == base:
            return prefix
        try:
            return f"{prefix}/{resolved.relative_to(base)}"
        except ValueError:
            continue
    return str(resolved)


def _print_audit(cfg: "Config") -> int:
    """Print the two-section audit; return the number of missing critical paths."""
    critical, generated = _audit_rows(cfg)

    def show(rows):
        key_w = max(len(label) for label, _, _ in rows)
        path_w = max(len(_shorten(cfg, r)) for _, _, r in rows)
        missing = 0
        for label, note, resolved in rows:
            if resolved is None:
                status = "UNRESOLVED"
                missing += 1
            elif resolved.exists():
                status = "OK"
            else:
                status = "MISSING"
                missing += 1
            print(f"  {status:<10} {label:<{key_w}}  "
                  f"{_shorten(cfg, resolved):<{path_w}}  {note}")
        return missing

    print(f"config file : {cfg.config_path}")
    print(f"<repo>      : {cfg.repo_root}")
    print(f"<data_root> : {cfg.data_root}")

    print()
    print("CRITICAL — inputs the pipeline reads and never regenerates")
    print("-" * 78)
    n_missing = show(critical)

    print()
    print("NON-CRITICAL — intermediates written by a pipeline step;")
    print("               MISSING only means that step has not run yet")
    print("-" * 78)
    n_pending = show(generated)

    print()
    if n_missing:
        print(f"{n_missing} critical input(s) missing — the pipeline cannot run "
              f"until these exist.")
    else:
        print("All critical inputs present.")
    print(f"{n_pending} of {len(generated)} intermediates not yet generated.")
    return n_missing


if __name__ == "__main__":
    import sys

    # Exits non-zero when a critical input is missing, so this doubles as a
    # pre-flight check in a shell script.
    sys.exit(1 if _print_audit(load_config()) else 0)
