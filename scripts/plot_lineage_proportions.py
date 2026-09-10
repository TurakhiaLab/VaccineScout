#!/usr/bin/env python3
"""
Monthly proportion of circulating Pango lineages from a MAT, in the style of
Figure 1a of the SARS-CoV-2 recombination paper.

Haplotype (leaf) names in an UShER MAT look like
``USA/CA-QDX-2597/2020|MW191321.1|2020-03-16``; the collection date is the last
``|``-delimited field. Leaves carrying an internal-node identifier (``node_17124``)
are not haplotypes and are skipped. Each haplotype's lineage is the Pango label of
its nearest annotated ancestor (``MATNode.most_recent_annotation()`` index 1, the
same convention as ``lineage_spike_from_mat.py``). Labels that UShER disambiguated
with a node id (``XBB.1.5_17124``) are folded into their parent (``XBB.1.5``); the
counts TSV keeps the original labels, so the merge happens at plotting time.

Lineages whose spike proteins are identical are drawn as a single segment with
their proportions combined, since no vaccine can tell them apart. The sets come
from the ``lineage_group:`` lines of the medoid assignment logs
(``--spike-groups``). Every member is combined however small. Such a segment
carries two names: inside the bar just its leading lineage with a star ("BA.4*"),
which keeps the label short, and in the legend the full "BA.4/BA.5" spelling out
every member that reaches 1% on its own.

For every month in the requested window the script stacks the proportion of each
lineage, keeps the most predominant lineages at the bottom, names every segment it
draws, and folds everything at or below ``--other-threshold`` into "Others".

Every lineage over that threshold gets its own color, dozens of them over a
multi-year window. Since eight hues is the readable ceiling for categorical color,
the two dimensions are split: **hue names the variant group** -- the WHO variants
of concern, then the antigenic groups FDA's VRBPAC has drawn vaccine composition
from (BA.4/BA.5, XBB, JN.1) -- and **lightness separates the lineages inside a
group**, so everything one vaccine was built against reads as shades of one color.

More groups than hues would mean handing a hue on once its group died out, which
``assign_family_hues`` does; with the groups this coarse each simply keeps its own.

Inside a group the lineages that clear ``--label-threshold`` are placed first and
get the spectrum to themselves, spaced evenly from its lightest shade to its
darkest, so the ones a reader picks out of a bar are as far apart as that hue can
put them. The remaining lineages are then spread evenly through the gaps left
between them, close to their neighbours but never on the same shade. Which
prominent lineage takes which of those shades is searched rather than assigned in
order, maximising the distance between the ones that share a bar.

Each 1 June is ruled off with a dotted line carried below the month labels and
named with that season's FDA-recommended vaccine strain, read from
``export_scout_medoid_report.py``. In the legend, the prominent lineages -- those
clearing ``--label-threshold`` -- are underlined.

Reading the tree and drawing the figure happen in one run. The `vaccinescout`
env carries both ``bte`` and ``matplotlib``, and every path defaults to
``config.yaml``, so the whole thing is:

    conda activate vaccinescout
    python scripts/plot_lineage_proportions.py

which reads ``files.mat`` and writes ``<paths.data_root>/lineage_proportions.svg``.

Extracting the counts from the MAT is the slow half and depends only on the tree,
so ``--counts-out`` caches it and ``--counts-in`` replots from that cache without
touching the MAT again -- useful when iterating on thresholds or colors.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from math import atan2, ceil, cos, degrees, radians, sin
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config_loader import load_config  # noqa: E402

CFG = load_config()

# --- from config.yaml ------------------------------------------------------
DEFAULT_MAT = CFG.file("mat")
# The plot covers the same window the rest of the pipeline is trimmed to, so it
# reads analysis.start_date/end_date rather than restating them.
DEFAULT_START = str(CFG.get("analysis", "start_date"))
DEFAULT_END = str(CFG.get("analysis", "end_date"))
# Index into a node's clade annotations holding the Pango label. Shared with the
# other MAT readers, so it lives in mat_scripts rather than being restated here.
DEFAULT_ANNOTATION_INDEX = int(CFG.get("mat_scripts", "annotation_index", 1))
# Logs whose "lineage_group:" lines record which lineages share a spike protein.
# vaccinescout_results_across_years.medoid_lineage_log_path() names them after
# the weighting exponent they were selected under, so the same analysis.beta
# that produced them picks them back up here.
SPIKE_GROUP_BETA = CFG.beta
SPIKE_GROUP_LOGS_PATTERN = (
    f"*/medoid_lineage_assignments_linear_exp_{SPIKE_GROUP_BETA:g}.log"
)
SPIKE_GROUP_LOGS_GLOB = str(CFG.data_root / SPIKE_GROUP_LOGS_PATTERN)
SPIKE_GROUP_LOGS_DEFAULT = sorted(CFG.data_root.glob(SPIKE_GROUP_LOGS_PATTERN))

# --- fixed here ------------------------------------------------------------
DEFAULT_OUTPUT = CFG.data_root / "lineage_proportions_across_years.svg"
DEFAULT_MAX_COLORS = None
DEFAULT_THEME = "light"
DEFAULT_LABEL_ROTATION = "auto"
# Drawn above the plot. --title "" turns it off.
DEFAULT_TITLE = "Monthly distribution of circulating SARS-CoV-2 lineages"
# Percent at or above which a lineage counts as prominent, and at or below which
# it is folded into "Others".
DEFAULT_LABEL_THRESHOLD = 10.0
DEFAULT_OTHER_THRESHOLD = 5.0
# In-bar label sizing. The plot is made tall enough that even the shallowest
# segment carrying a name reaches TARGET; roomier segments grow up to MAX, and
# FLOOR is the last resort before a name is dropped as unfittable. FLOOR is set
# where type stops being worth reading rather than where it stops being drawable:
# a shallow segment loses its name to the legend instead of carrying it in 8pt.
LABEL_FLOOR_POINTS = 6.0
LABEL_TARGET_POINTS = 8.0
LABEL_MAX_POINTS = 22.0
# Width:height the figure is held to. Width grows a fixed amount per month while
# the label rule below only asks for enough height to set a turned name in the
# shallowest labelled segment -- about 10in whatever the window. So the longer the
# window the flatter the figure: five years is 47.6in wide, which against that 10in
# read as nearly 5:1. Height is therefore taken from the width as well, and the
# figure gets whichever of the two rules asks for more.
FIGURE_TARGET_ASPECT = 2.6
# Figure height the point sizes further down were chosen against. Everything that
# is not an in-bar label is set in absolute points, so on a figure taller than this
# it reads proportionally smaller -- the type has to grow with the height the way
# the height now grows with the width. In-bar labels are exempt: _label_fontsize
# derives them from segment geometry, which already follows the figure.
FIGURE_REFERENCE_HEIGHT_INCHES = 10.0
# Ceiling on that growth. Height now follows width, so on a long window the scale
# would otherwise run away with it -- at 68 months it reached 2.72, putting the
# y-ticks at 76pt while the dates, whose size is capped by their own month's pitch
# and cannot follow, stayed near 30. Past this point the figure grows and the type
# holds, which is the same trade the dates already make.
TYPE_SCALE_MAX = 1.6
# Ceiling on the height either rule may ask for, so neither one nor an unusually
# long name can stretch the figure past what a page or screen can show.
FIGURE_HEIGHT_MAX_INCHES = 26.0
# Width the figure gives each month, and the share of it the bar fills -- the
# remainder is the gap that keeps adjacent months apart.
MONTH_WIDTH_INCHES = 0.95
BAR_WIDTH_SHARE = 0.82
# Percent a lineage must reach on its own to be spelled out in the legend name of
# the combined segment it belongs to. Lower than the bar for being drawn at all,
# since a lineage too small to plot on its own can still be worth naming among the
# handful it shares a spike with.
SPIKE_NAME_THRESHOLD = 1.0
# The legend's type size and depth -- the entries spread sideways into as many
# columns as those rows need.
TICK_LABEL_POINTS = 28
LEGEND_FONT_POINTS = 0.8 * TICK_LABEL_POINTS - 2.0
LEGEND_ROWS = 4
# The two headings, in the same points-at-reference-height terms as the rest. Both
# sit above the tick labels in the hierarchy: in-bar <= 22, legend 22.4, ticks 28,
# axis title 34.1, figure title 39.6.
# The month labels are the one piece of type sized from the figure's width rather
# than its height: the window grows by adding months, so what constrains a date is
# the pitch its own month gets, which follows MONTH_WIDTH_INCHES and nothing else.
# _fit_date_labels measures the drawn row and solves for the largest size holding
# the ratio below, so widening the figure enlarges the dates on its own.
#
# The ratio is of bounding-box width to pitch, and it sits above 1 on purpose: the
# labels are rotated 45 degrees, so their boxes may overlap well past touching
# while the glyphs stay on clear diagonals. 1.8 is the measured point at which
# every month is still separately readable; 4.0 was the smear this started as.
DATE_LABEL_OVERLAP = 1.8
DATE_LABEL_TRIAL_POINTS = 20.0
# Size a date is worth reading at, as a share of the y-tick size, and the most
# months that may be skipped to reach it.
#
# Widening the figure cannot get there. A date's room is the pitch, pitch is
# width / months, so its size relative to the whole figure goes as 1 / months and
# MONTH_WIDTH_INCHES cancels out -- 0.7in and 1.2in per month give the same
# apparent size once the figure is on screen whole. Labelling every second or
# third month is what actually enlarges them, because it is the only change that
# gives one label more than one month's width.
DATE_LABEL_TARGET_SHARE = 0.8
DATE_LABEL_MAX_STRIDE = 3
# Never larger than the y-axis ticks, so the busiest type is never the loudest.
# Measured against the scaled tick size, not the raw constant.
Y_LABEL_POINTS = 1.55 * LABEL_MAX_POINTS
TITLE_POINTS = 1.8 * LABEL_MAX_POINTS
# Columns the inline block uses. None derives them from LEGEND_ROWS; a number asks
# for that many directly, and the rows then follow from the entry count -- the two
# cannot be set independently, since entries = rows x columns.
LEGEND_COLUMNS: Optional[int] = 11
# Widest the inline block goes, as a backstop on a run of very many lineages. It
# is deliberately above what LEGEND_ROWS asks for, so the row count is what
# normally decides the shape; _fit_legend still drops columns that do not fit.
LEGEND_MAX_COLUMNS = 12
# Lineages a combined name may list before the entry is moved to a row of its own,
# and the gap in axes fractions between the plot's foot and the legend.
LEGEND_INLINE_PARTS = 3
LEGEND_GAP = 0.015
# Legend geometry, each in multiples of the legend's own type size: the space
# between adjacent entries on a row, and the pad the block holds inside its own
# edges. Both are named because the long-form block has to work out how many
# names a row takes before it draws the row.
LEGEND_COLUMN_SPACING = 1.4
LEGEND_BORDER_PAD = 0.4
LEGEND_LABEL_SPACING = 0.6
LEGEND_AXES_PAD = 0.5
# Share of the figure's width a legend block may take before it wraps onto
# another row rather than running past the plot.
LEGEND_WIDTH_SHARE = 0.98
OTHER_LABEL = "Others"
UNASSIGNED_LABEL = "Unassigned"

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
# Internal-node identifiers (e.g. "node_17124") are not haplotypes; a MAT that
# carries them as leaves must not contribute them as samples.
INTERNAL_NODE_RE = re.compile(r"^node_\d+$", re.IGNORECASE)
# UShER annotates a Pango lineage on several nodes when it is not monophyletic,
# disambiguating each with a node id ("XBB.1.5_17124"). Those are all the parent
# lineage. Only a purely numeric suffix qualifies, so labels such as
# "XBB.1_27915T" — which name a mutation, not a node — are left alone.
NODE_SUFFIX_RE = re.compile(r"_\d+$")

# Colors are a composite encoding: hue names the variant group, lightness separates
# the lineages inside it. Dozens of lineages clear the threshold over four years,
# and generating dozens of hues would be unreadable (eight is the categorical
# ceiling), so the validated 8-hue order below is spent on groups -- handed out in
# fixed order, never cycled -- and each group's lineages take shades of its hue.
FAMILY_SLOTS = {
    "light": [
        "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
        "#e87ba4", "#008300", "#4a3aa7", "#e34948",
    ],
    "dark": [
        "#3987e5", "#d95926", "#199e70", "#c98500",
        "#d55181", "#008300", "#9085e9", "#e66767",
    ],
}

# Clade families in the order they emerged, so the families that co-circulate in a
# month -- the ones that end up touching in a stack -- land on adjacent palette
# slots, which the validated order guarantees are separable.
#
# Groups follow how the strains have actually been treated rather than the finest
# available taxonomy: the WHO variants of concern, then the antigenic groups FDA's
# VRBPAC has picked vaccine composition from -- BA.4/BA.5 together for the 2022-23
# bivalent, the XBB lineage for 2023-24, the JN.1 lineage (KP.2, then LP.8.1) for
# 2024-26. Splitting these into their two dozen Nextstrain clades scatters what a
# reader thinks of as one variant across unrelated hues.
#
#
# Membership is by Pango alias prefix. Pango aliases are hierarchical and cannot be
# resolved without the alias key, so this is a curated table: it covers every alias
# that has held a meaningful share to date, and anything unrecognised is reported by
# name at run time so a new alias can be added here.
CLADE_FAMILIES: List[Tuple[str, Tuple[str, ...]]] = [
    # Wuhan-Hu-1 and its early descendants -- what the original vaccine was built
    # on. The bare "A"/"B" prefixes only ever match lineages with no more specific
    # rule, since the longest prefix wins.
    ("Ancestral (original vaccine)", ("A", "B")),
    # The WHO variants of concern and of interest, each its own group. Every one
    # of these is a B.* or P.*/C.* lineage, so without a rule of its own the
    # longest-prefix match drops it into "Ancestral" above -- which for Beta, a
    # variant defined by escape from ancestral-strain immunity, is backwards.
    # Only Mu has an alias descending from it (BB = B.1.621.1); the rest are
    # matched by their bare Pango name, verified against pango-designation's
    # alias_key.json.
    ("Epsilon", ("B.1.427", "B.1.429")),
    ("Zeta", ("P.2",)),
    ("Alpha", ("B.1.1.7", "Q")),
    ("Beta", ("B.1.351",)),
    ("Gamma", ("P.1",)),
    ("Theta", ("P.3",)),
    ("Eta", ("B.1.525",)),
    ("Iota", ("B.1.526",)),
    ("Kappa", ("B.1.617.1",)),
    ("Lambda", ("C.37",)),
    ("Mu", ("B.1.621", "BB")),
    ("Delta", ("B.1.617.2", "AY")),
    (
        "Omicron BA.1 / BA.2",
        ("BA.1", "BC", "BD", "BA.2", "BG", "BH", "BJ", "BP", "BS", "BY", "CA", "CB",
         "BL", "BM", "BN", "BR", "CH", "CJ", "CV", "DV"),
    ),
    (
        "BA.4 / BA.5 (2022-23 vaccine)",
        ("BA.4", "CS", "DC", "BA.5", "BE", "BF", "BK", "BQ", "BT", "BU", "BV", "BW",
         "CD", "CE", "CF", "CK", "CL", "CN", "CP", "CQ", "CR", "CT", "CY", "DL"),
    ),
    (
        "XBB (2023-24 vaccine)",
        ("XBB", "EG", "EK", "EL", "EM", "EU", "FD", "FE", "FG", "FL", "FP", "FT",
         "FU", "FY", "GC", "GD", "GE", "GF", "GJ", "GK", "GN", "GS", "GV", "GY",
         "HF", "HG", "HK", "HN", "HV", "HZ", "JA", "JC", "JD", "JF", "JG"),
    ),
    (
        "JN.1 (2024-26 vaccine)",
        ("BA.2.86", "JN", "KP", "KQ", "KS", "KZ", "LB", "LF", "LK", "LP", "MA",
         "MB", "MC", "MK", "MV", "NB", "XDV", "XEC", "XFG"),
    ),
]

# The FDA-recommended vaccine strain per season, marked at the June boundary that
# opens it. These come from the `vaccines:` block of config.yaml, the same source
# vaccinescout_results_across_years.py reads them from -- taken directly rather
# than by importing that module, which would drag in pandas/pyarrow/Bio to read
# four strings. A config without the block still draws the boundaries, unnamed.
def fda_strains_by_cutoff() -> Dict[str, str]:
    try:
        strains = CFG.section("vaccines")["fda_strain_by_cutoff"]
    except (KeyError, ValueError) as exc:
        print(f"Note: FDA strain names unavailable ({exc}); drawing unnamed boundaries.")
        return {}
    named = {str(cutoff): str(strain) for cutoff, strain in strains.items()}
    named.update(EXTRA_FDA_STRAINS)
    return named


# The 2022-23 booster was bivalent -- BA.4/BA.5 alongside the ancestral strain --
# so its marker names both components; the other seasons were monovalent.
FDA_CO_STRAIN = {"2022-06-01": "B"}

# Seasons to rule off that vaccines.fda_strain_by_cutoff does not carry. That key
# doubles as the cutoff list for vaccinescout_results_across_years.py
# (CUTOFF_DATES_DEFAULT), so a season with no cohort built on disk cannot be added
# there without asking that script to analyse a cutoff it has no data for. The
# 2025-26 formula is monovalent JN.1-lineage, preferentially LP.8.1 (VRBPAC,
# 22 May 2025).
EXTRA_FDA_STRAINS = {"2025-06-01": "LP.8.1"}

# How far a hue's shades run either side of its own documented lightness, and the
# hard limits that keeps them inside so the ends stay clear of the surface.
#
# The range has to follow the hue rather than be shared by all of them. A single
# band drags every hue to the same lightness, and a hue pulled far from its own
# stops reading as itself: yellow sits at L 0.76, so a band ending at 0.38 turned
# it into brown, which is also where dark orange lands. Centring each hue on its
# own step keeps a group's shades recognisably one colour.
# The spread is asymmetric, reaching further into tints than into shades. Darkening
# is what costs a hue its identity -- a dark orange or yellow is brown, and browns
# from two different hues look alike -- while lightening only ever makes a paler
# version of the same colour. Reaching up rather than down keeps a group's
# gradient recognisably one hue.
RAMP_SPREAD_UP = 0.18
RAMP_SPREAD_DOWN = 0.10
RAMP_LIMITS = {"light": (0.34, 0.86), "dark": (0.44, 0.90)}
# Lightness gap two shades of one hue need before they read as different colors.
MIN_STEP_DELTA_L = 0.06
# Hand nudges along a group's gradient, applied after the automatic placement.
# Positive is darker, negative lighter, measured as a fraction of the group's own
# band. The allocation only ever reasons about lineages sharing a hue, so it
# cannot see two lineages from different groups whose shades happen to land close;
# nudging one of them is cheaper than re-tuning the whole scheme around one pair.
# Names are as they appear in the bars, with or without the combined-segment star.
SHADE_NUDGES = {"AY.100": 0.15}
# Restarts of the shade-ordering search, and its seed. A family is small, so this
# is cheap; the seed keeps a given dataset rendering identically every run.
SHADE_SEARCH_RESTARTS = 400
SHADE_SEARCH_SEED = 20210501

THEME = {
    "light": {
        "surface": "#fcfcfb",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "other": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "other": "#52514e",
    },
}


# --------------------------------------------------------------------------
# color assignment: hue per clade family, lightness per lineage
# --------------------------------------------------------------------------

def family_of(lineage: str) -> Optional[str]:
    """The clade a Pango label belongs to, or None if unrecognised.

    The longest matching prefix wins, so a clade carved out of a broader one --
    KP.3.1.1 out of KP.3, BA.2.12.1 out of BA.2 -- is matched by its own rule
    however the table happens to be ordered. Names combining lineages that share a
    spike ("BA.4/BA.5", or "BA.4*" in a bar) resolve on whichever part is
    recognised; they share a spike, so they cannot straddle two groups.
    """
    best: Optional[Tuple[int, str]] = None
    for part in lineage.rstrip("*").split("/"):
        for family, prefixes in CLADE_FAMILIES:
            for prefix in prefixes:
                if part == prefix or part.startswith(prefix + "."):
                    if best is None or len(prefix) > best[0]:
                        best = (len(prefix), family)
    return best[1] if best else None


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055


def hex_to_oklch(hex_color: str) -> Tuple[float, float, float]:
    """sRGB hex -> OKLCH (L 0..1, C, hue in degrees)."""
    r, g, b = (_srgb_to_linear(int(hex_color[i : i + 2], 16) / 255) for i in (1, 3, 5))
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)

    lab_l = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s
    lab_a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
    lab_b = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s

    chroma = (lab_a**2 + lab_b**2) ** 0.5
    hue = degrees(atan2(lab_b, lab_a)) % 360.0
    return lab_l, chroma, hue


def _oklch_to_rgb(lightness: float, chroma: float, hue: float) -> Tuple[float, float, float]:
    lab_a = chroma * cos(radians(hue))
    lab_b = chroma * sin(radians(hue))

    l = (lightness + 0.3963377774 * lab_a + 0.2158037573 * lab_b) ** 3
    m = (lightness - 0.1055613458 * lab_a - 0.0638541728 * lab_b) ** 3
    s = (lightness - 0.0894841775 * lab_a - 1.2914855480 * lab_b) ** 3

    return (
        _linear_to_srgb(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
        _linear_to_srgb(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
        _linear_to_srgb(-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s),
    )


def oklch_to_hex(lightness: float, chroma: float, hue: float) -> str:
    """OKLCH -> sRGB hex, reducing chroma until the color fits the sRGB gamut.

    Holding L and the hue and giving up only chroma keeps a ramp monotone in
    lightness and single-hue, which is what an ordinal ramp is checked on.
    """
    low, high = 0.0, chroma
    for _ in range(24):
        mid = (low + high) / 2
        if all(-1e-4 <= channel <= 1 + 1e-4 for channel in _oklch_to_rgb(lightness, mid, hue)):
            low = mid
        else:
            high = mid
    rgb = _oklch_to_rgb(lightness, low, hue)
    return "#" + "".join(f"{round(255 * min(max(c, 0.0), 1.0)):02x}" for c in rgb)


def _closest_shared_bar_gap(
    members: Sequence[str],
    windows: Dict[str, Tuple[str, str]],
    positions: Dict[str, float],
    prominent: Sequence[str],
    base_hex: str,
    theme: str,
) -> Optional[float]:
    """Lightness between the closest two named shades of a group sharing a month.

    Only the lineages carrying a name are measured. The minor ones are packed into
    the room the named ones leave over, so they are meant to sit closer together;
    warning about that would be warning about the design.
    """
    named = set(prominent)
    lo, hi = ramp_band(base_hex, theme)
    gaps = [
        abs(positions[one] - positions[other]) * (hi - lo)
        for index, one in enumerate(members)
        for other in members[index + 1 :]
        if one in named and other in named and _windows_overlap(windows, one, other)
    ]
    return min(gaps) if gaps else None


def ramp_band(base_hex: str, theme: str) -> Tuple[float, float]:
    """Lightness range a hue's shades span, centred on its documented step."""
    base_l, _chroma, _hue = hex_to_oklch(base_hex)
    low_limit, high_limit = RAMP_LIMITS[theme]
    return (
        max(low_limit, base_l - RAMP_SPREAD_DOWN),
        min(high_limit, base_l + RAMP_SPREAD_UP),
    )


def shade_at(base_hex: str, position: float, theme: str) -> str:
    """The hue's shade at ``position`` along its band, 0 lightest to 1 darkest."""
    _base_l, chroma, hue = hex_to_oklch(base_hex)
    lo, hi = ramp_band(base_hex, theme)
    return oklch_to_hex(hi - position * (hi - lo), chroma, hue)


def _windows_overlap(
    windows: Dict[str, Tuple[str, str]], one: str, other: str
) -> bool:
    """Whether two lineages are ever on screen in the same month."""
    start_a, end_a = windows[one]
    start_b, end_b = windows[other]
    return start_a <= end_b and start_b <= end_a


def assign_family_hues(
    grouped: Dict[str, List[str]], windows: Dict[str, Tuple[str, str]]
) -> Tuple[Dict[str, int], List[str]]:
    """Give every clade on the chart one of the eight hues.

    There are far more clades than hues -- Nextstrain has named a dozen inside the
    JN.1 lineage alone -- but only a handful circulate at any one time, and only
    those ever have to be told apart. So a hue is handed on once the clade holding
    it has died out: clades that share a month always get different hues, and two
    clades a year apart may share one, which is invisible to a reader comparing
    what is on screen. Hues go out in the validated order, so the clades competing
    in one month land on slots that order guarantees are separable.

    Returns the hue slot per clade, plus a note for any month that ran more clades
    at once than there are hues.
    """
    spans = {
        family: (
            min(windows[member][0] for member in members),
            max(windows[member][1] for member in members),
        )
        for family, members in grouped.items()
    }
    order = sorted(spans, key=lambda f: (spans[f][0], f))

    hue_of: Dict[str, int] = {}
    slot_free_from: List[str] = []  # per hue, the month its last clade left
    load: Counter = Counter()  # lineages already wearing each hue
    over_subscribed: List[str] = []

    for family in order:
        start, end = spans[family]
        # Among the hues going spare, take the one carrying the fewest lineages.
        # Every lineage on a hue costs the others some of its band, so letting the
        # early hues soak up clade after clade is what squeezes the shades.
        candidates = [slot for slot, free in enumerate(slot_free_from) if free < start]
        if len(slot_free_from) < len(FAMILY_SLOTS["light"]):
            candidates.append(len(slot_free_from))  # a hue nobody has worn yet
        if candidates:
            # The least-worn hue going spare. Every lineage on a hue costs the
            # others part of its band, so an untouched hue (load zero) is taken
            # ahead of one already carrying a clade, and letting the early hues
            # soak up clade after clade -- which is what squeezes the shades --
            # never happens while any hue is lighter.
            slot = min(candidates, key=lambda s: (load[s], s))
            if slot == len(slot_free_from):
                slot_free_from.append(end)
            else:
                slot_free_from[slot] = end
        else:
            # Out of hues: reuse the one whose clade left longest ago, which is
            # the least likely to still be on screen beside this one.
            slot = min(range(len(slot_free_from)), key=lambda s: slot_free_from[s])
            slot_free_from[slot] = max(slot_free_from[slot], end)
            over_subscribed.append(family)
        hue_of[family] = slot
        load[slot] += len(grouped[family])

    if over_subscribed:
        over_subscribed = [
            f"more clades circulate at once than the {len(FAMILY_SLOTS['light'])} "
            f"hues can separate, so {', '.join(over_subscribed)} reuse a hue"
        ]
    return hue_of, over_subscribed


def _even_positions(count: int) -> List[float]:
    """``count`` shades spread over the whole band, lightest to darkest."""
    if count <= 0:
        return []
    if count == 1:
        return [0.0]
    return [index / (count - 1) for index in range(count)]


def _gap_positions(anchors: Sequence[float], count: int) -> List[float]:
    """``count`` shades spread through what the anchors leave free.

    The prominent lineages are placed first and hold the ends of the band, so the
    room left over is the runs between them; this shares the remaining lineages
    out across those runs as evenly as they divide.
    """
    if count <= 0:
        return []
    if len(anchors) < 2:
        # One anchor sits at the light end (or there are none at all), so the
        # leftover room is simply the rest of the band.
        start = 0.0 if not anchors else anchors[0]
        return [start + (1.0 - start) * (index + 1) / count for index in range(count)]

    runs = list(zip(anchors, anchors[1:]))
    share, extra = divmod(count, len(runs))
    positions: List[float] = []
    for index, (low, high) in enumerate(runs):
        here = share + (1 if index < extra else 0)
        step = (high - low) / (here + 1)
        positions.extend(low + step * (offset + 1) for offset in range(here))
    return positions


def assign_shade_positions(
    members: Sequence[str],
    windows: Dict[str, Tuple[str, str]],
    prominent: Sequence[str],
) -> Dict[str, float]:
    """Place each member of a family along its clade's gradient, 0 light to 1 dark.

    The lineages that clear the label threshold are the ones a reader picks out of
    a bar and looks up, so they get the spectrum to themselves: however many of
    them a group has, they are spaced evenly from the lightest shade to the
    darkest, which is the furthest apart that hue can put them. Only then are the
    minor lineages placed, evenly through the gaps left between them -- so a
    lineage nobody reads off the chart never costs a prominent one its separation,
    and no two lineages land on the same shade either way.

    Which prominent lineage takes which of those evenly spaced shades still
    matters, and that is searched: positions are dealt out repeatedly, each
    lineage taking the free one furthest from everything it shares a month with,
    and the best arrangement over the restarts is kept. Pairs that are both named
    on the chart are what the search optimises first; the rest break ties. A group
    holds at most a couple of dozen lineages, so the search costs milliseconds.
    """
    if len(members) == 1:
        return {members[0]: 0.0}

    prominent_set = set(prominent) & set(members)
    major = [m for m in members if m in prominent_set]
    minor = [m for m in members if m not in prominent_set]
    major_slots = _even_positions(len(major))
    minor_slots = _gap_positions(major_slots, len(minor))

    overlapping = [
        (one, other)
        for index, one in enumerate(members)
        for other in members[index + 1 :]
        if _windows_overlap(windows, one, other)
    ]
    headline = [pair for pair in overlapping if set(pair) <= prominent_set]

    def score(slots: Dict[str, float]) -> Tuple[List[float], List[float]]:
        return (
            sorted(abs(slots[a] - slots[b]) for a, b in headline),
            sorted(abs(slots[a] - slots[b]) for a, b in overlapping),
        )

    conflicts = defaultdict(set)
    for one, other in overlapping:
        conflicts[one].add(other)
        conflicts[other].add(one)

    chronological = sorted(members, key=lambda l: (windows[l][0], l))
    # Seeded so a given dataset always yields the same figure.
    rng = random.Random(SHADE_SEARCH_SEED)
    best_slots: Optional[Dict[str, float]] = None
    best_score: Optional[Tuple[List[float], List[float]]] = None

    for attempt in range(SHADE_SEARCH_RESTARTS):
        order = chronological if attempt == 0 else rng.sample(chronological, len(members))
        slots: Dict[str, float] = {}
        free = {True: list(major_slots), False: list(minor_slots)}
        for lineage in order:
            pool = free[lineage in prominent_set]
            rivals = [slots[other] for other in conflicts[lineage] if other in slots]
            named = [
                slots[other]
                for other in conflicts[lineage]
                if other in slots and lineage in prominent_set and other in prominent_set
            ]
            if rivals:
                slot = max(
                    pool,
                    key=lambda s: (
                        min((abs(s - r) for r in named), default=1.0),
                        min(abs(s - r) for r in rivals),
                        -s,
                    ),
                )
            else:
                slot = min(pool)
            slots[lineage] = slot
            pool.remove(slot)

        attempt_score = score(slots)
        if best_score is None or attempt_score > best_score:
            best_slots, best_score = slots, attempt_score

    return best_slots


def assign_colors(
    colored: Sequence[str],
    shares: Dict[str, Dict[str, float]],
    months: Sequence[str],
    threshold: float,
    label_threshold: float,
    theme: str,
) -> Tuple[Dict[str, str], List[str], List[str], List[str]]:
    """Map every colored lineage to a hex.

    Returns the color map, the lineages in legend order (grouped by family, each
    family in emergence order), any lineages whose family is unrecognised -- those
    keep the "Others" gray, since inventing a hue for them is exactly the thing
    this scheme avoids -- and the families whose peak concurrency outran the ramp,
    whose shades are therefore closer than ``MIN_STEP_DELTA_L``.
    """
    windows: Dict[str, Tuple[str, str]] = {}
    prominent: List[str] = []
    for month in months:
        for lineage, share in shares.get(month, {}).items():
            if share <= threshold:
                continue
            start, _end = windows.get(lineage, (month, month))
            windows[lineage] = (start, month)
            if share >= label_threshold and lineage not in prominent:
                prominent.append(lineage)

    grouped: Dict[str, List[str]] = defaultdict(list)
    unclassified: List[str] = []
    for lineage in colored:
        family = family_of(lineage)
        if family is None or lineage not in windows:
            unclassified.append(lineage)
        else:
            grouped[family].append(lineage)

    color_of: Dict[str, str] = {}
    legend_order: List[str] = []
    crowded: List[str] = []
    slots = FAMILY_SLOTS[theme]
    hue_of, over_subscribed = assign_family_hues(grouped, windows)

    # Shades are dealt per hue rather than per clade. A hue is shared by clades
    # from different eras, and giving each of them the same shades would hand two
    # lineages the identical color; pooling them means every lineage wearing a hue
    # gets a shade of it that no other lineage has.
    by_slot: Dict[int, List[str]] = defaultdict(list)
    for family, members in grouped.items():
        by_slot[hue_of[family]].extend(members)

    positions: Dict[str, float] = {}
    for slot, members in by_slot.items():
        placed = assign_shade_positions(members, windows, prominent)
        # Hand nudges land before the gap is measured, so the warning reports the
        # shades actually drawn -- a nudge spends separation inside the group to
        # buy it against another, and that has to show up in the check.
        for lineage in placed:
            nudge = SHADE_NUDGES.get(lineage) or SHADE_NUDGES.get(lineage.rstrip("*"))
            if nudge:
                placed[lineage] = min(1.0, max(0.0, placed[lineage] + nudge))
        positions.update(placed)
        # What matters is not how many shades the hue holds but how far apart the
        # ones that share a bar ended up, so measure that pair.
        gap = _closest_shared_bar_gap(
            members, windows, placed, prominent, slots[slot], theme
        )
        if gap is not None and gap < MIN_STEP_DELTA_L:
            clades = sorted({family_of(member) for member in members})
            crowded.append(f"{'/'.join(clades)} (ΔL {gap:.3f})")

    for family, _prefixes in CLADE_FAMILIES:
        members = grouped.get(family)
        if not members:
            continue
        slot = hue_of[family]
        # A hue carrying a single lineage keeps the documented palette hex.
        for lineage in sorted(members, key=lambda l: (windows[l][0], l)):
            color_of[lineage] = (
                slots[slot]
                if len(by_slot[slot]) == 1
                else shade_at(slots[slot], positions[lineage], theme)
            )
            legend_order.append(lineage)

    if over_subscribed:
        crowded.extend(over_subscribed)

    return color_of, legend_order, sorted(unclassified), crowded


# --------------------------------------------------------------------------
# MAT -> per-month lineage counts
# --------------------------------------------------------------------------

def parse_sample_date(name: str) -> Optional[str]:
    """Return the YYYY-MM-DD collection date encoded in a haplotype name."""
    tail = name.rsplit("|", 1)[-1].strip()
    if DATE_RE.fullmatch(tail):
        return tail
    match = DATE_RE.search(name)
    return match.group(0) if match else None


def count_lineages_by_month(
    mat_path: Path,
    start: str,
    end: str,
    annotation_index: int = 1,
) -> Tuple[Counter, Dict[str, int]]:
    """Count leaves per (month, lineage) for haplotypes collected within [start, end].

    Lineage labels are taken verbatim from the tree, so UShER's per-node
    disambiguation (``XBB.1.5_17124``) stays its own lineage.
    """
    import bte  # imported lazily: only the extraction half needs it

    tree = bte.MATree(str(mat_path))
    counts: Counter = Counter()
    stats = {"total": 0, "not_haplotype": 0, "no_date": 0, "out_of_window": 0, "kept": 0}

    for node in tree.get_leaves():
        stats["total"] += 1
        if INTERNAL_NODE_RE.match(node.id.strip()):
            stats["not_haplotype"] += 1
            continue
        sample_date = parse_sample_date(node.id)
        if sample_date is None:
            stats["no_date"] += 1
            continue
        if not (start <= sample_date <= end):
            stats["out_of_window"] += 1
            continue

        annotations = node.most_recent_annotation()
        label = annotations[annotation_index] if len(annotations) > annotation_index else ""
        counts[(sample_date[:7], label.strip() or UNASSIGNED_LABEL)] += 1
        stats["kept"] += 1

    return counts, stats


def month_range(start: str, end: str) -> List[str]:
    """Every YYYY-MM touched by [start, end], in order."""
    first = datetime.strptime(start, "%Y-%m-%d").date().replace(day=1)
    last = datetime.strptime(end, "%Y-%m-%d").date().replace(day=1)
    months: List[str] = []
    cursor = first
    while cursor <= last:
        months.append(cursor.strftime("%Y-%m"))
        cursor = date(cursor.year + cursor.month // 12, cursor.month % 12 + 1, 1)
    return months


def write_counts(counts: Counter, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["month", "lineage", "count"])
        for (month, lineage), count in sorted(counts.items()):
            writer.writerow([month, lineage, count])
    print(f"Wrote {path}")


def read_counts(path: Path) -> Counter:
    counts: Counter = Counter()
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            counts[(row["month"], row["lineage"])] += int(row["count"])
    return counts


# --------------------------------------------------------------------------
# counts -> stacked proportions
# --------------------------------------------------------------------------

def merge_node_suffixed_lineages(counts: Counter) -> Tuple[Counter, Dict[str, str]]:
    """Fold "<lineage>_<node id>" labels into the parent lineage they belong to.

    Returns the merged counts and, for reporting, a map of every label that moved
    to the parent it was merged into.
    """
    merged: Counter = Counter()
    moves: Dict[str, str] = {}

    for (month, lineage), count in counts.items():
        parent = lineage
        # Labels can stack suffixes ("BA.1_22877_22878"); strip until stable so the
        # result is a real Pango name rather than a shorter node-suffixed one.
        while NODE_SUFFIX_RE.search(parent):
            parent = NODE_SUFFIX_RE.sub("", parent)
        if parent != lineage:
            moves[lineage] = parent
        merged[(month, parent)] += count

    return merged, moves


def load_spike_groups(paths: Sequence[Path]) -> Dict[str, Tuple[str, ...]]:
    """Map each lineage to the lineages carrying an identical spike protein.

    Read from the ``lineage_group:`` lines of the medoid assignment logs, where a
    group is the set of Pango lineages whose spike sequences are the same. A
    vaccine cannot tell those apart, so the chart should not either.
    """
    membership: Dict[str, Tuple[str, ...]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if "lineage_group:" not in line:
                continue
            members = tuple(sorted(line.split("lineage_group:", 1)[1].strip().split("|")))
            for member in members:
                membership[member] = members
    return membership


def merge_spike_groups(
    counts: Counter,
    months: Sequence[str],
    membership: Dict[str, Tuple[str, ...]],
    threshold: float,
) -> Tuple[Counter, Dict[str, str]]:
    """Combine lineages sharing a spike into one segment, under two names.

    A group can run to a hundred-odd lineages, nearly all of them slivers. Every
    member's counts are combined, and the result gets a short name for the bar and
    a full one for the legend: the bar carries only the leading lineage with a
    star ("BA.4*"), which keeps the label short enough not to drive the figure's
    size, while the legend spells out every member that reaches ``threshold`` on
    its own ("BA.4/BA.5"). A group whose members only clear it together is named
    for its largest.

    Returns the combined counts and, for the segments that are combinations, the
    full name to show in the legend.
    """
    shares = monthly_proportions(counts, months)
    peak: Dict[str, float] = defaultdict(float)
    for month_shares in shares.values():
        for lineage, share in month_shares.items():
            peak[lineage] = max(peak[lineage], share)

    vaccine_strains = set(fda_strains_by_cutoff().values())
    short_of: Dict[Tuple[str, ...], str] = {}
    legend_names: Dict[str, str] = {}
    merged: Counter = Counter()

    for (month, lineage), count in counts.items():
        members = membership.get(lineage)
        if members is None or len(members) == 1:
            merged[(month, lineage)] += count
            continue
        if members not in short_of:
            shown = sorted(member for member in members if peak[member] >= threshold)
            if not shown:
                shown = [max(members, key=lambda m: (peak[m], m))]
            # A group holding a strain FDA built a vaccine against goes by that
            # strain: the group is what the vaccine covers, and XBB.1.5 is what a
            # reader is looking for, not whichever member sorts first.
            shown.sort(key=lambda m: (m not in vaccine_strains, m))
            short_of[members] = f"{shown[0]}*"
            legend_names[short_of[members]] = "/".join(shown)
        merged[(month, short_of[members])] += count

    return merged, legend_names


def monthly_proportions(counts: Counter, months: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Per month, each lineage's percentage of that month's samples."""
    totals: Counter = Counter()
    for (month, _lineage), count in counts.items():
        totals[month] += count

    shares: Dict[str, Dict[str, float]] = {}
    for month in months:
        total = totals.get(month, 0)
        if not total:
            shares[month] = {}
            continue
        shares[month] = {
            lineage: 100.0 * count / total
            for (m, lineage), count in counts.items()
            if m == month
        }
    return shares


def select_colored_lineages(
    shares: Dict[str, Dict[str, float]],
    threshold: float,
    label_threshold: float,
    max_colors: Optional[int],
) -> List[str]:
    """Choose which lineages get their own color and stack segment.

    Candidates are lineages that exceed ``threshold`` in at least one month; by
    default every one of them is kept, since the family ramps supply as many
    shades as a family has members. ``max_colors`` optionally caps the count, and
    then the ranking matters: by how many months a lineage carries a name (its
    share clears ``label_threshold``), then by its peak monthly share.
    """
    peak: Dict[str, float] = defaultdict(float)
    named: Counter = Counter()

    for month_shares in shares.values():
        for lineage, share in month_shares.items():
            if share <= threshold:
                continue
            peak[lineage] = max(peak[lineage], share)
            if share >= label_threshold:
                named[lineage] += 1

    ranked = sorted(peak, key=lambda l: (-named[l], -peak[l], l))
    return ranked if max_colors is None else ranked[:max_colors]


def build_stacks(
    shares: Dict[str, Dict[str, float]],
    colored: Sequence[str],
    threshold: float,
    label_threshold: float,
) -> Dict[str, List[Tuple[str, float, bool]]]:
    """Per month, the stack from bottom to top as (lineage, share, is_prominent).

    A lineage gets its own segment when it has a color and exceeds the threshold
    that month; everything at or below it lands in "Others". The most predominant
    lineage sits at the bottom and "Others" always sits on top. Every segment
    carries its name; the flag marks the ones holding at least ``label_threshold``
    percent of the month, which are the ones the figure is sized around, the ones
    underlined in the legend, and the ones given the run of their hue's spectrum.
    """
    colored_set = set(colored)
    stacks: Dict[str, List[Tuple[str, float, bool]]] = {}

    for month, month_shares in shares.items():
        segments = sorted(
            (
                (share, lineage)
                for lineage, share in month_shares.items()
                if lineage in colored_set and share > threshold
            ),
            reverse=True,
        )
        stack = [(lineage, share, share >= label_threshold) for share, lineage in segments]

        other = 100.0 - sum(share for _l, share, _f in stack)
        if other > 1e-9:
            stack.append((OTHER_LABEL, other, False))
        stacks[month] = stack

    return stacks


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------

def _ink_on(hex_color: str) -> str:
    """Black or white label ink, whichever reads on the given fill."""
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))

    def channel(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    luminance = 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
    return "#0b0b0b" if luminance > 0.45 else "#ffffff"


def _label_fontsize(
    label: str,
    along_points: float,
    across_points: float,
    base: float = LABEL_MAX_POINTS,
    floor: float = LABEL_FLOOR_POINTS,
) -> Optional[float]:
    """Shrink a segment label until it fits, since lineage names run long.

    ``along_points`` is the room the text runs into (the bar's width when
    horizontal, the segment's height when turned on its side) and
    ``across_points`` the room for its line height. None when even the floor size
    would not fit, so a name is dropped rather than spilled over its neighbours.
    """
    if not label:
        return base
    fitted = min(0.95 * along_points / (0.6 * len(label)), 0.9 * across_points)
    return None if fitted < floor else min(base, fitted)


def _measure_date_row(fig, ax) -> Tuple[float, float]:
    """(widest label, pitch) of the month labels as currently drawn, in pixels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in ax.get_xticklabels()]
    if len(boxes) < 2:
        return 0.0, 0.0
    corners = sorted(box.x0 for box in boxes)
    return (
        max(box.width for box in boxes),
        (corners[-1] - corners[0]) / (len(corners) - 1),
    )


def _fit_date_labels(fig, ax, months: Sequence[str], max_points: float) -> float:
    """Set the month labels as large as they can be read, thinning if need be.

    Measures the row as drawn -- at DATE_LABEL_TRIAL_POINTS -- because the space a
    label has is set by the axes' width and the month count, neither known until
    the figure exists. Width scales linearly with point size, so one measurement
    gives the size at any stride: skipping months multiplies the pitch and nothing
    else. The smallest stride reaching the target wins, so a short window keeps
    every month and only a long one starts dropping them.
    """
    widest, pitch = _measure_date_row(fig, ax)
    if widest <= 0 or pitch <= 0:
        return DATE_LABEL_TRIAL_POINTS

    def size_at(stride: int) -> float:
        return min(
            DATE_LABEL_TRIAL_POINTS * DATE_LABEL_OVERLAP * pitch * stride / widest,
            max_points,
        )

    target = max_points * DATE_LABEL_TARGET_SHARE
    stride = next(
        (s for s in range(1, DATE_LABEL_MAX_STRIDE + 1) if size_at(s) >= target),
        DATE_LABEL_MAX_STRIDE,
    )
    if stride > 1:
        kept = list(range(0, len(months), stride))
        ax.set_xticks(kept)
        ax.set_xticklabels(
            [datetime.strptime(months[i], "%Y-%m").strftime("%b %Y") for i in kept],
            rotation=45,
            ha="right",
        )
        print(
            f"Labelling every {stride}{'nd' if stride == 2 else 'rd'} month: "
            f"{len(months)} months at {size_at(1):.0f}pt would be unreadable, "
            f"{len(kept)} at {size_at(stride):.0f}pt is not."
        )
    return size_at(stride)


def _lowest_axes_y(ax, artists) -> float:
    """The lowest point the given artists reach, in axes fraction."""
    inverse = ax.transAxes.inverted()
    return min(
        inverse.transform((0, artist.get_window_extent().y0))[1] for artist in artists
    )


def _draw_season_boundaries(
    fig, ax, months: Sequence[str], ink: dict, tick_points: float = TICK_LABEL_POINTS
) -> float:
    """Rule off each vaccine season at 1 June and name the FDA strain for it.

    The rule is dotted, runs the height of the plot and carries on past the month
    labels so the strain name sits clear of them. Returns the axes-fraction depth
    everything below the axis now reaches, which is where the legend goes next.
    """
    fig.canvas.draw()
    foot = _lowest_axes_y(ax, ax.get_xticklabels())

    strains = fda_strains_by_cutoff()
    # Only Junes with a named strain get a boundary. Every other June in the
    # window -- 2020, before the earliest cutoff -- would otherwise draw a rule
    # with nothing to say, since the label lookup below can come up empty.
    junes = [
        (idx, month)
        for idx, month in enumerate(months)
        if month.endswith("-06") and strains.get(f"{month}-01")
    ]
    if not junes:
        return foot

    for idx, month in junes:
        # The boundary sits between May and June rather than on June itself.
        ax.plot(
            [idx - 0.5, idx - 0.5],
            [1.0, foot - 0.02],
            transform=ax.get_xaxis_transform(),
            linestyle=(0, (4, 3)),
            color=ink["text_primary"],
            linewidth=2.6,
            clip_on=False,
            zorder=5,
        )
        strain = strains[f"{month}-01"]
        partner = FDA_CO_STRAIN.get(f"{month}-01")
        ax.text(
            idx - 0.5,
            foot - 0.04,
            f"FDA: {strain} + {partner}" if partner else f"FDA: {strain}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=tick_points,
            fontweight="bold",
            color=ink["text_secondary"],
            clip_on=False,
        )

    fig.canvas.draw()
    return _lowest_axes_y(ax, ax.get_xticklabels() + ax.texts)


def _underline_legend_entries(
    fig,
    legends: Sequence,
    names: set,
    ink: dict,
    label_to_key: Optional[Dict[str, str]] = None,
) -> None:
    """Rule under the legend entries that are named in the bars.

    Matplotlib has no underlined text, so the rules are drawn from the laid-out
    text boxes -- which also keeps them exactly as wide as the name they mark.

    Every legend block is passed in, not just the inline one: a combined segment
    earns its rule the same way an ordinary lineage does, and its entry lives in
    the long-form block underneath. Those entries also read as their spelled-out
    membership rather than the short key the bars use, so ``label_to_key`` maps
    the drawn text back to the lineage that ``names`` is keyed by.
    """
    from matplotlib.lines import Line2D

    fig.canvas.draw()
    inverse = fig.transFigure.inverted()
    lookup = label_to_key or {}

    for legend in legends:
        if legend is None:
            continue
        for text in legend.get_texts():
            label = text.get_text()
            if lookup.get(label, label) not in names:
                continue
            box = text.get_window_extent()
            (x0, y0), (x1, _y1) = inverse.transform([(box.x0, box.y0), (box.x1, box.y1)])
            # Hold the rule a slice of the type size below the baseline, so it
            # tracks its own text. A fixed figure fraction drifts as the type
            # grows until it reads as a line struck through the row beneath.
            drop = 0.12 * text.get_fontsize() / 72 / fig.get_figheight()
            fig.add_artist(
                Line2D(
                    [x0, x1],
                    [y0 - drop, y0 - drop],
                    transform=fig.transFigure,
                    color=ink["text_secondary"],
                    linewidth=0.9,
                )
            )


def plot_proportions(
    stacks: Dict[str, List[Tuple[str, float, bool]]],
    months: Sequence[str],
    color_of: Dict[str, str],
    legend_order: Sequence[str],
    output_path: Path,
    theme: str = "light",
    threshold: float = DEFAULT_OTHER_THRESHOLD,
    title: str = "",
    label_rotation: str = "auto",
    legend_names: Optional[Dict[str, str]] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ink = THEME[theme]
    color_of = dict(color_of)
    color_of[OTHER_LABEL] = ink["other"]

    # Widen with the window so bars stay wide enough to hold a lineage name and the
    # month ticks keep their spacing; a year or two still gets the compact figure.
    fig, ax = plt.subplots(
        figsize=(max(18.0, MONTH_WIDTH_INCHES * len(months)), 5.0)
    )
    fig.patch.set_facecolor(ink["surface"])
    ax.set_facecolor(ink["surface"])
    fig.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.15)

    # Wide bars so lineage names fit at full size; the remainder is the gap that
    # keeps adjacent months visually separate.
    bar_fraction = BAR_WIDTH_SHARE
    bar_points = (
        ax.get_position().width * fig.get_figwidth() / max(len(months), 1) * bar_fraction * 72
    )
    # Every segment carries its name; the prominent ones are what the figure is
    # sized around, and the rest take whatever room that leaves.
    prominent = [
        (lineage, share)
        for month in months
        for lineage, share, is_prominent in stacks.get(month, [])
        if is_prominent
    ]
    widest = max(
        (
            len(lineage)
            for month in months
            for lineage, _share, _prominent in stacks.get(month, [])
            if lineage != OTHER_LABEL
        ),
        default=1,
    )

    if label_rotation == "auto":
        # Names run along the bar until the bars get too narrow to hold them --
        # over a multi-year window that happens well before the segments run out
        # of height, so the labels turn on their side instead.
        label_rotation = (
            "vertical"
            if 0.95 * bar_points / (0.6 * widest) < LABEL_FLOOR_POINTS
            else "horizontal"
        )
    vertical = label_rotation == "vertical"

    # Hold the figure to its target aspect. This applies however the labels are
    # set, since a long window is shallow-looking whether or not its names turn.
    heights = [fig.get_figheight(), fig.get_figwidth() / FIGURE_TARGET_ASPECT]

    if vertical and prominent:
        # A turned label runs along its segment's height, so the shallowest
        # segment that must carry a name sets how tall the figure has to be. Size
        # that height for the target rather than the floor, so the names come out
        # readable instead of merely present. It is sized on the prominent
        # segments: sizing on the smallest sliver on the chart would double the
        # figure's height for the sake of a lineage nobody is looking up.
        shallowest = min(share for _lineage, share in prominent)
        fraction = ax.get_position().height
        heights.append(
            LABEL_TARGET_POINTS * 0.6 * widest / (0.95 * shallowest * fraction * 0.72)
        )

    fig.set_figheight(min(max(heights), FIGURE_HEIGHT_MAX_INCHES))

    # Every point size below is quoted at FIGURE_REFERENCE_HEIGHT_INCHES and scaled
    # to the height the figure actually ended up at, so the type keeps its share of
    # the figure however long the window is. Never scaled down: a short window keeps
    # the sizes the constants name outright.
    type_scale = min(
        max(1.0, fig.get_figheight() / FIGURE_REFERENCE_HEIGHT_INCHES), TYPE_SCALE_MAX
    )
    tick_points = TICK_LABEL_POINTS * type_scale
    legend_points = LEGEND_FONT_POINTS * type_scale
    ylabel_points = Y_LABEL_POINTS * type_scale
    title_points = TITLE_POINTS * type_scale

    # Points of height one percentage point of a stack is worth, which is the room
    # a turned label runs into.
    point_per_percent = ax.get_position().height * fig.get_figheight() * 72 / 100.0

    x = range(len(months))
    for idx, month in enumerate(months):
        bottom = 0.0
        for lineage, share, _is_prominent in stacks.get(month, []):
            fill = color_of.get(lineage, ink["other"])
            ax.bar(
                idx,
                share,
                bottom=bottom,
                width=bar_fraction,
                color=fill,
                edgecolor=ink["surface"],
                linewidth=1.5,
                zorder=2,
            )
            segment_points = share * point_per_percent
            fontsize = _label_fontsize(
                lineage,
                segment_points if vertical else bar_points,
                bar_points if vertical else segment_points,
            )
            if lineage != OTHER_LABEL and fontsize is not None:
                ax.text(
                    idx,
                    bottom + share / 2,
                    lineage,
                    ha="center",
                    va="center",
                    rotation=90 if vertical else 0,
                    fontsize=fontsize,
                    color=_ink_on(fill),
                    zorder=3,
                )
            bottom += share

    # Bars run to the edges of the plot: matplotlib's default margins would leave
    # a bar's width of dead space before the first month and after the last.
    ax.set_xlim(-0.5, len(months) - 0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [datetime.strptime(m, "%Y-%m").strftime("%b %Y") for m in months],
        rotation=45,
        ha="right",
        fontsize=DATE_LABEL_TRIAL_POINTS,
    )
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)], fontsize=tick_points)
    # The y-axis title runs the height of the plot, and the figure is deliberately
    # wide and shallow, so it is sized to that height rather than set outright --
    # a fixed size overflows the axis and gets clipped once the plot is short.
    y_title = "Percentage of circulating variants"
    ax.set_ylabel(
        y_title,
        fontsize=ylabel_points,
        color=ink["text_secondary"],
    )
    if title:
        ax.set_title(
            title,
            fontsize=title_points,
            fontweight="bold",
            color=ink["text_primary"],
            pad=14,
        )

    ax.set_axisbelow(True)
    ax.grid(axis="y", color=ink["grid"], linewidth=1, zorder=0)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ink["axis"])
    # Split by axis: a single tick_params call sets labelsize on both, which
    # silently overrode the size set_xticklabels had just given the dates.
    ax.tick_params(axis="x", colors=ink["muted"], labelsize=_fit_date_labels(fig, ax, months, tick_points))
    ax.tick_params(axis="y", colors=ink["muted"], labelsize=tick_points)

    foot = _draw_season_boundaries(fig, ax, months, ink, tick_points)

    # Legend order follows the color assignment -- families in emergence order,
    # each family's shades light to dark -- so it reads as the epidemic's timeline
    # and the shades of a hue sit together.
    # A combined segment goes by its short name in the bars and its full one here,
    # which is the only place there is room to spell the membership out.
    spelled = legend_names or {}
    handles = [
        Patch(facecolor=color_of[l], label=spelled.get(l, l)) for l in legend_order
    ]
    handles.append(Patch(facecolor=color_of[OTHER_LABEL], label=f"{OTHER_LABEL} (≤{threshold:g}%)"))

    # A name listing more than LEGEND_INLINE_PARTS lineages is several times the
    # width of an ordinary one and would set the column width for the whole grid,
    # so those go in their own block underneath.
    long_form = [h for h in handles if h.get_label().count("/") + 1 > LEGEND_INLINE_PARTS]
    inline = [h for h in handles if h not in long_form]

    def _draw_legend(entries, anchor, columns):
        drawn = ax.legend(
            handles=entries,
            loc="upper center",
            bbox_to_anchor=(0.5, anchor),
            ncol=columns,
            frameon=False,
            fontsize=legend_points,
            handlelength=1.4,
            handleheight=1.1,
            columnspacing=LEGEND_COLUMN_SPACING,
            borderpad=LEGEND_BORDER_PAD,
            borderaxespad=LEGEND_AXES_PAD,
            labelspacing=LEGEND_LABEL_SPACING,
        )
        for text in drawn.get_texts():
            text.set_color(ink["text_secondary"])
        return drawn

    def _fit_legend(entries, anchor, wanted_columns):
        """Draw the entries, dropping columns until the block fits the figure.

        A fixed row count cannot hold: however many rows are asked for, enough
        entries -- or one long enough name -- make the block wider than the plot
        it sits under. So the row count gives way instead, and the block wraps
        onto as many rows as the width allows.

        The test is where the block's edges actually land, not how wide it is.
        The legend is centred on the axes, and the axes sits right of centre in
        the figure because the y-label and tick labels take room on the left --
        so a block narrow enough to "fit" can still hang off the right edge and
        lose its last entry.
        """
        span = fig.get_figwidth() * fig.dpi
        margin = (1 - LEGEND_WIDTH_SHARE) / 2 * span
        left_limit, right_limit = margin, span - margin

        columns = max(1, wanted_columns)
        drawn = _draw_legend(entries, anchor, columns)
        while columns > 1:
            fig.canvas.draw()
            box = drawn.get_window_extent()
            if box.x0 >= left_limit and box.x1 <= right_limit:
                break
            # Width tracks the column count closely enough to jump most of the
            # way in one step; the -1 guarantees the loop makes progress.
            allowed = right_limit - left_limit
            columns = min(columns - 1, max(1, int(columns * allowed / box.width)))
            drawn.remove()
            drawn = _draw_legend(entries, anchor, columns)
        return drawn

    def _row_capacity():
        """Pixels a legend row has, given that it is centred on the axes.

        The block is centred on the plot, and the plot sits right of the figure's
        centre because the y-label and tick labels take room on the left -- so the
        room a row really has is twice its shorter side, not the whole width.
        """
        span = fig.get_figwidth() * fig.dpi
        margin = (1 - LEGEND_WIDTH_SHARE) / 2 * span
        box = ax.get_position()
        centre = (box.x0 + box.x1) / 2 * span
        return 2 * min(centre - margin, span - margin - centre)

    def _row_step():
        """Axes fraction to drop between one legend row and the next.

        Rows of a single legend are separated by ``labelspacing`` alone, but rows
        drawn as separate legends -- which is how the long-form block fills each
        row to its own width -- are held apart by three pads besides: the border pad
        each block keeps inside its own edge, and the pad the lower one keeps from
        the point it is anchored to. Left alone those space the rows half again as
        far apart as the block above. Taking all three back off leaves both blocks
        reading as one evenly spaced list; the result is negative, since the pads
        come to more than the spacing, so the next row is pulled up, not dropped.
        """
        axes_points = ax.get_position().height * fig.get_figheight() * 72
        pads = 2 * LEGEND_BORDER_PAD + LEGEND_AXES_PAD
        return (LEGEND_LABEL_SPACING - pads) * legend_points / axes_points

    def _entry_widths(entries, anchor):
        """How wide each entry comes out, in pixels, measured one at a time.

        Only a laid-out legend knows how wide a name renders, and the rows have to
        be filled before they can be drawn, so each entry is drawn alone, measured
        and thrown away, with the block's own padding taken back off.
        """
        pad = 2 * LEGEND_BORDER_PAD * legend_points * fig.dpi / 72
        widths = []
        for entry in entries:
            probe = _draw_legend([entry], anchor, 1)
            fig.canvas.draw()
            widths.append(probe.get_window_extent().width - pad)
            probe.remove()
        return widths

    def _pack_legend(entries, anchor):
        """Draw the entries in rows filled to the width, rather than in a grid.

        A combined name spelling out its membership runs several times the width
        of an ordinary one, and matplotlib's ``ncol`` grid gives every column the
        width of the longest name in it and the same number of rows -- so a block
        mixing one long name with shorter ones pads the short ones out to the long
        one's width and leaves the last row nearly empty. Filling each row with
        whatever fits keeps the block as few rows deep as the names allow.

        Returns the legends drawn, top row first.
        """
        allowed = _row_capacity()
        pad = 2 * LEGEND_BORDER_PAD * legend_points * fig.dpi / 72
        gap = LEGEND_COLUMN_SPACING * legend_points * fig.dpi / 72

        rows: List[list] = []
        row: list = []
        used = pad
        for entry, width in zip(entries, _entry_widths(entries, anchor)):
            step = width + (gap if row else 0.0)
            if row and used + step > allowed:
                rows.append(row)
                row, used, step = [], pad, width
            row.append(entry)
            used += step
        if row:
            rows.append(row)

        drawn = []
        for row in rows:
            if drawn:
                # ax.legend() replaces the axes' legend, so the row above has to be
                # parked as an ordinary artist before the next one takes its place.
                ax.add_artist(drawn[-1])
            drawn.append(_fit_legend(row, anchor, len(row)))
            fig.canvas.draw()
            anchor = _lowest_axes_y(ax, [drawn[-1]]) - _row_step()
        return drawn

    # Spread the entries sideways rather than stacking them in a narrow block: a
    # few roomy rows of many columns, so the names stay large enough to read.
    anchor = foot - LEGEND_GAP
    legend = None
    long_legends: List = []
    if inline:
        wanted = LEGEND_COLUMNS or ceil(len(inline) / LEGEND_ROWS)
        legend = _fit_legend(inline, anchor, min(LEGEND_MAX_COLUMNS, wanted))

    if long_form:
        if legend is not None:
            ax.add_artist(legend)
            fig.canvas.draw()
            anchor = _lowest_axes_y(ax, [legend]) - _row_step()
        long_legends = _pack_legend(long_form, anchor)

    named = {
        lineage
        for month in months
        for lineage, _share, is_prominent in stacks.get(month, [])
        if is_prominent
    }
    # A merged group is drawn under its short key but listed under its spelled-out
    # membership, so the rule has to be matched back through that renaming.
    label_to_key = {spelled.get(l, l): l for l in legend_order}
    _underline_legend_entries(fig, [legend, *long_legends], named, ink, label_to_key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Format follows the extension, which is what --help advertises; hardcoding
    # svg here wrote SVG bytes into any name the caller gave, .png included.
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Wrote {output_path}")


def report_prominent_lineages(
    shares: Dict[str, Dict[str, float]],
    months: Sequence[str],
    threshold: float,
    label_threshold: float,
) -> None:
    """List every lineage that ever reached ``label_threshold``, and how long it lasted.

    Peak share says how big a lineage got; the month count says how long it was
    worth drawing. The two answer different questions -- a lineage can spike for a
    single month or hold a tenth of the population for a year.
    """
    peak: Dict[str, float] = defaultdict(float)
    peak_month: Dict[str, str] = {}
    circulating: Counter = Counter()
    first_seen: Dict[str, str] = {}

    for month in months:
        for lineage, share in shares.get(month, {}).items():
            if share >= threshold:
                circulating[lineage] += 1
                first_seen.setdefault(lineage, month)
            if share > peak[lineage]:
                peak[lineage], peak_month[lineage] = share, month

    prominent = sorted(
        (l for l, top in peak.items() if top >= label_threshold),
        key=lambda l: (first_seen.get(l, peak_month[l]), l),
    )
    if not prominent:
        print(f"No lineage reached {label_threshold:g}% in any month.")
        return

    print(
        f"\n{len(prominent)} lineage(s) reached {label_threshold:g}% in at least one "
        f"month, with the months they held >={threshold:g}%:"
    )
    width = max(len(lineage) for lineage in prominent)
    print(
        f"  {'Lineage':<{width}}  {'Group':<30}  {'Peak %':>7}  "
        f"{'Peak':<8}  {'Months >=' + format(threshold, 'g') + '%':>13}"
    )
    for lineage in prominent:
        print(
            f"  {lineage:<{width}}  {family_of(lineage) or '(unclassified)':<30}  "
            f"{peak[lineage]:>7.1f}  {peak_month[lineage]:<8}  {circulating[lineage]:>13}"
        )
    print()


def write_table(
    stacks: Dict[str, List[Tuple[str, float, bool]]],
    months: Sequence[str],
    path: Path,
) -> None:
    """The plotted numbers, so identity never rests on color alone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["month", "stack_position", "lineage", "percent", "prominent"])
        for month in months:
            for position, (lineage, share, is_prominent) in enumerate(stacks.get(month, []), 1):
                writer.writerow([month, position, lineage, f"{share:.3f}", int(is_prominent)])
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the monthly proportion of Pango lineages among MAT haplotypes, "
            "naming the top lineages each month and folding the tail into 'Others'."
        )
    )
    parser.add_argument(
        "--mat",
        type=Path,
        default=None,
        help=f"MAT protobuf (.pb / .pb.gz). Requires bte. Default: files.mat ({DEFAULT_MAT}).",
    )
    parser.add_argument(
        "--counts-in",
        type=Path,
        help="Skip the MAT and read a month/lineage/count TSV written by --counts-out.",
    )
    parser.add_argument(
        "--counts-out",
        type=Path,
        help="Cache the per-month lineage counts to this TSV, to replot with --counts-in.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Figure path (.png/.svg/.pdf). Default: {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--table-out",
        type=Path,
        help="Write the plotted per-segment percentages to this TSV.",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help=f"Inclusive start date (default analysis.start_date, {DEFAULT_START}).",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_END,
        help=f"Inclusive end date (default analysis.end_date, {DEFAULT_END}).",
    )
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=DEFAULT_LABEL_THRESHOLD,
        help=(
            "Percent at or above which a lineage counts as prominent "
            f"(default {DEFAULT_LABEL_THRESHOLD:g}). Every segment is named; these "
            "are the ones the figure is sized around, underlined in the legend, "
            "and given the run of their group's spectrum."
        ),
    )
    parser.add_argument(
        "--other-threshold",
        type=float,
        default=DEFAULT_OTHER_THRESHOLD,
        help=(
            "Percent at or below which a lineage joins 'Others' "
            f"(default {DEFAULT_OTHER_THRESHOLD:g})."
        ),
    )
    parser.add_argument(
        "--max-colors",
        type=int,
        default=DEFAULT_MAX_COLORS,
        help=(
            "Cap the lineages given their own color/segment; the rest join 'Others'. "
            "By default every lineage over the threshold is colored, since each "
            "clade family supplies as many shades as it has members."
        ),
    )
    parser.add_argument(
        "--annotation-index",
        type=int,
        default=DEFAULT_ANNOTATION_INDEX,
        help=(
            "Index into a node's clade annotations for the Pango label "
            f"(default mat_scripts.annotation_index, {DEFAULT_ANNOTATION_INDEX})."
        ),
    )
    parser.add_argument(
        "--spike-groups",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Medoid assignment logs whose 'lineage_group:' lines list the lineages "
            "sharing a spike protein; those are drawn as one segment named "
            f"'BA.4/BA.5'. Defaults to {SPIKE_GROUP_LOGS_GLOB}, named after "
            f"analysis.beta ({SPIKE_GROUP_BETA:g}). "
            "Pass with no paths to keep every lineage separate."
        ),
    )
    parser.add_argument(
        "--theme",
        choices=("light", "dark"),
        default=DEFAULT_THEME,
        help=f"Figure theme (default {DEFAULT_THEME}).",
    )
    parser.add_argument(
        "--label-rotation",
        choices=("auto", "horizontal", "vertical"),
        default=DEFAULT_LABEL_ROTATION,
        help=(
            "Orientation of the lineage names drawn inside the bars. 'auto' turns "
            "them on their side once the bars are too narrow to hold them "
            f"(default {DEFAULT_LABEL_ROTATION})."
        ),
    )
    parser.add_argument(
        "--title",
        default=DEFAULT_TITLE,
        help=f"Figure title; pass an empty string for none (default {DEFAULT_TITLE!r}).",
    )
    args = parser.parse_args()

    if args.mat and args.counts_in:
        parser.error("--mat and --counts-in are mutually exclusive")
    if not args.counts_in:
        # Both halves run here, so the tree is the default input rather than
        # something the caller has to name.
        args.mat = args.mat or DEFAULT_MAT
        if not args.mat.is_file():
            parser.error(
                f"MAT not found: {args.mat}. Set files.mat in config.yaml, or pass "
                "--mat, or --counts-in to replot cached counts."
            )
    if not 0 < args.label_threshold <= 100:
        parser.error("--label-threshold must be between 0 and 100")
    if args.label_threshold < args.other_threshold:
        parser.error("--label-threshold must not be below --other-threshold")
    if args.max_colors is not None and args.max_colors < 1:
        parser.error("--max-colors must be at least 1")
    for value in (args.start, args.end):
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            parser.error(f"dates must be YYYY-MM-DD, got {value!r}")
    if args.start > args.end:
        parser.error("--start must not be after --end")

    if args.counts_in:
        counts = read_counts(args.counts_in)
    else:
        counts, stats = count_lineages_by_month(
            args.mat,
            args.start,
            args.end,
            annotation_index=args.annotation_index,
        )
        print(
            f"{stats['total']} leaves: kept {stats['kept']} in "
            f"[{args.start}, {args.end}], {stats['out_of_window']} outside the window, "
            f"{stats['no_date']} without a parseable date, "
            f"{stats['not_haplotype']} internal-node names skipped"
        )
        if args.counts_out:
            write_counts(counts, args.counts_out)

    if not counts:
        print("No samples in the requested window.", file=sys.stderr)
        raise SystemExit(1)

    counts, moves = merge_node_suffixed_lineages(counts)
    if moves:
        print(
            f"Merged {len(moves)} node-suffixed label(s) into their parent lineage: "
            + ", ".join(f"{child} -> {parent}" for child, parent in sorted(moves.items()))
        )

    months = [m for m in month_range(args.start, args.end) if any(mm == m for mm, _ in counts)]

    group_paths = (
        [path for path in SPIKE_GROUP_LOGS_DEFAULT if path.is_file()]
        if args.spike_groups is None
        else args.spike_groups
    )
    legend_names: Dict[str, str] = {}
    if group_paths:
        membership = load_spike_groups(group_paths)
        counts, legend_names = merge_spike_groups(
            counts, months, membership, SPIKE_NAME_THRESHOLD
        )
        if legend_names:
            print(
                f"Combined lineages sharing a spike into {len(legend_names)} segment(s) "
                f"from {len(group_paths)} log(s): "
                + ", ".join(
                    f"{short} = {full}" for short, full in sorted(legend_names.items())
                )
            )
    elif args.spike_groups is None:
        print(
            "Note: no spike-group logs found "
            f"({SPIKE_GROUP_LOGS_GLOB}); every lineage is drawn separately."
        )

    shares = monthly_proportions(counts, months)
    colored = select_colored_lineages(
        shares, args.other_threshold, args.label_threshold, args.max_colors
    )
    stacks = build_stacks(shares, colored, args.other_threshold, args.label_threshold)

    report_prominent_lineages(shares, months, args.other_threshold, args.label_threshold)

    color_of, legend_order, unclassified, crowded = assign_colors(
        colored, shares, months, args.other_threshold, args.label_threshold, args.theme
    )
    families = Counter(family_of(lineage) for lineage in legend_order)
    print(
        f"Colored {len(legend_order)} lineage(s) across {len(families)} clade "
        "familie(s): "
        + ", ".join(f"{family} ({count})" for family, count in families.items())
    )
    if crowded:
        print(
            "Note: too many lineages circulate at once for one hue to separate them "
            f"in {', '.join(crowded)} -- the closest pair sharing a bar falls under "
            f"the ΔL {MIN_STEP_DELTA_L} the eye resolves reliably, so read those off "
            "the segment labels or --table-out rather than color alone."
        )

    if unclassified:
        # Without a family there is no hue to shade, and inventing one is the very
        # thing this scheme avoids -- so these stay in "Others" until CLADE_FAMILIES
        # learns their alias.
        colored = [lineage for lineage in colored if lineage not in set(unclassified)]
        stacks = build_stacks(shares, colored, args.other_threshold, args.label_threshold)
        print(
            f"Note: {len(unclassified)} lineage(s) have no entry in CLADE_FAMILIES and "
            f"joined '{OTHER_LABEL}'; add their alias prefix to color them: "
            + ", ".join(unclassified)
        )

    dropped = {
        lineage
        for month_shares in shares.values()
        for lineage, share in month_shares.items()
        if share > args.other_threshold and lineage not in set(colored)
    }
    dropped -= set(unclassified)
    if dropped:
        print(
            f"Note: {len(dropped)} lineage(s) exceeded {args.other_threshold:g}% but exceeded "
            f"--max-colors and joined '{OTHER_LABEL}': {', '.join(sorted(dropped))}"
        )

    if args.table_out:
        write_table(stacks, months, args.table_out)
    if args.output:
        plot_proportions(
            stacks,
            months,
            color_of,
            legend_order,
            args.output,
            theme=args.theme,
            threshold=args.other_threshold,
            title=args.title,
            label_rotation=args.label_rotation,
            legend_names=legend_names,
        )


if __name__ == "__main__":
    main()
