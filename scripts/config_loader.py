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

    def cohort_file(self, cutoff_date: str, key: str, create: bool = False) -> Path:
        """
        A ``files:`` entry that belongs to one cutoff, not to the data root.

        The circulating cohort is rebuilt for every ``cutoff_date``, so files
        describing it (pairwise distances, the id->sequence map, the mutation
        sites) are per-cutoff. They resolve to 
        ``<data_root>/<cutoff_date>/<filename>``.
        """
        return self.cutoff_dir(cutoff_date, create=create) / Path(
            str(self.section("files")[key])
        ).name

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
        # eve_subdir sits in analysis:, beside eve_backbone. fitness: is still
        # consulted so a config that put it there keeps resolving; the literal is
        # the last resort. Reading only fitness: used to work purely because the
        # default matched, and silently ignored a changed analysis.eve_subdir.
        subdir = (
            self.get("analysis", "eve_subdir")
            or self.get("fitness", "eve_subdir")
            or "EVEscape"
        )
        return self.data_root / str(subdir) / str(backbone)

    @property
    def fitness_column(self) -> str:
        return self.get("fitness", "fitness_column", "fitness_mean")

    # -- CoVFit -------------------------------------------------------------
    def covfit_outdir(self, cutoff_date: str, kind: str = "current",
                      create: bool = True) -> Path:
        """Output directory for a CoVFit run (``current``/``fda``/``pfizer``)."""
        key = f"{kind}_subdir"
        subdir = self.get("covfit", key)
        if subdir is None:
            known = sorted(k[:-len("_subdir")] for k in self.section("covfit")
                           if k.endswith("_subdir"))
            raise KeyError(
                f"covfit.{key} is not defined in {self.config_path}. "
                f"Configured kinds: {'|'.join(known) or 'none'}."
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
# The audit (`python scripts/config_loader.py`).
#
# It reports the pipeline's inputs only: the files you download, the third-party
# checkouts, and the binaries you build. Nothing regenerates these, so a missing
# one is a hard stop.
#
# Intermediates are deliberately not listed. Every step rewrites its own outputs,
# so their absence says nothing except "that step has not run yet" -- a wall of
# MISSING lines that buries the handful actually needing attention.
# ---------------------------------------------------------------------------

#: The downloaded input files, as ``(files: key, label)``. These four resolve
#: straight out of the ``files:`` block; the PyR0, EVEscape and Pfizer inputs are
#: keyed on run parameters instead and are added by ``_input_rows`` below.
INPUT_FILES = (
    ("mat",         "MAT tree (.pb)"),
    ("tree_newick", "Newick tree (.nwk)"),
    ("metadata",    "Metadata (.tsv)"),
    ("reference",   "Reference genome (.fa)"),
)

#: ``paths:`` keys that must already exist.
CRITICAL_PATHS = ("data_root", "pyro_cov", "covfit_cli", "pfizer_dir", "bin_dir")


def _input_rows(cfg: "Config"):
    """``(label, note, path)`` for every downloaded input the pipeline reads."""
    rows = []
    for key, label in INPUT_FILES:
        try:
            rows.append((label, f"files.{key}", cfg.file(key)))
        except Exception as exc:  # noqa: BLE001 - report, do not abort the audit
            rows.append((label, f"unresolved: {exc}", None))

    for label, note, getter in (
        ("PyR0 mutations (.tsv)",
         f"fitness.pyro_version={cfg.get('fitness', 'pyro_version')}",
         cfg.pyro_mutations_tsv),
        ("EVEscape scores",
         f"analysis.eve_backbone={cfg.get('analysis', 'eve_backbone')}",
         cfg.eve_scores),
    ):
        try:
            rows.append((label, note, getter()))
        except Exception as exc:  # noqa: BLE001
            rows.append((label, f"unresolved: {exc}", None))

    # One Pfizer genome per cutoff, not just the one for analysis.cutoff_date:
    # step 10 sweeps every cutoff, so all of them are inputs to a full run.
    try:
        by_cutoff = cfg.section("vaccines").get("pfizer_fasta_by_cutoff", {})
    except Exception as exc:  # noqa: BLE001
        rows.append(("Pfizer genome", f"unresolved: {exc}", None))
        by_cutoff = {}
    for cutoff in sorted(by_cutoff):
        try:
            rows.append((f"Pfizer genome {cutoff}",
                         "vaccines.pfizer_fasta_by_cutoff", cfg.pfizer_fasta(cutoff)))
        except Exception as exc:  # noqa: BLE001
            rows.append((f"Pfizer genome {cutoff}", f"unresolved: {exc}", None))
    return rows


def _support_rows(cfg: "Config"):
    """``(label, note, path)`` for the directories and binaries a run needs."""
    rows = []
    for key in CRITICAL_PATHS:
        try:
            rows.append((key, f"paths.{key}", cfg.path(key)))
        except Exception as exc:  # noqa: BLE001
            rows.append((key, f"unresolved: {exc}", None))

    # binary() raises when a helper is missing, which is exactly what an audit
    # must not do -- resolve the paths by hand instead.
    executable = str(cfg.get("covfit", "executable", "./covfit_cli"))
    rows.append(("covfit_cli", "covfit.executable",
                 cfg.path("covfit_cli") / Path(executable).name))
    for name in ("min_dist", "pairwise_dist"):
        rows.append((name, "built by cmake", cfg.path("bin_dir") / name))
    return rows


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
    """Print the critical-input audit; return the number that are missing."""
    inputs = _input_rows(cfg)
    support = _support_rows(cfg)

    key_w = max(len(label) for label, _, _ in inputs + support)
    path_w = max(len(_shorten(cfg, r)) for _, _, r in inputs + support)

    def show(rows):
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
    print("=" * 78)
    print("Input files you download")
    print("-" * 78)
    n_missing = show(inputs)

    print()
    print("Directories and executables")
    print("-" * 78)
    n_missing += show(support)

    print()
    if n_missing:
        print(f"{n_missing} critical input(s) missing — the pipeline cannot run "
              f"until these exist.")
    else:
        print("All critical inputs present.")
    return n_missing


if __name__ == "__main__":
    import sys

    # Exits non-zero when a critical input is missing, so this doubles as a
    # pre-flight check in a shell script.
    sys.exit(1 if _print_audit(load_config()) else 0)
