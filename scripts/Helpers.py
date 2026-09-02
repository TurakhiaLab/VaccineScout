import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from Bio import SeqIO
from typing import *

SPIKE_AA_SUB_RE = re.compile(r"^S:([A-Za-z*])(\d+)([A-Za-z*])")


def patch_numpy2_unpickle_compat() -> None:
    """Register NumPy 1.x aliases so pickles saved with NumPy 2.x can be loaded."""
    import importlib
    import numpy.core as _core

    sys.modules.setdefault("numpy._core", _core)

    for sub in ("numeric", "multiarray", "umath", "_multiarray_umath", "_dtype", "fromnumeric"):
        name = f"numpy._core.{sub}"
        if name in sys.modules:
            continue
        try:
            sys.modules[name] = importlib.import_module(f"numpy.core.{sub}")
        except ModuleNotFoundError:
            mod = getattr(_core, sub, None)
            if mod is not None:
                sys.modules[name] = mod

    if not any(type(f).__name__ == "_NumpyCoreCompatFinder" for f in sys.meta_path):
        class _NumpyCoreCompatFinder:
            """Map numpy._core.* imports to numpy.core.* on NumPy 1.x."""

            def find_module(self, fullname, path=None):
                if fullname == "numpy._core" or fullname.startswith("numpy._core."):
                    return self

            def load_module(self, fullname):
                if fullname in sys.modules:
                    return sys.modules[fullname]
                legacy = fullname.replace("numpy._core", "numpy.core", 1)
                mod = importlib.import_module(legacy)
                sys.modules[fullname] = mod
                return mod

        sys.meta_path.insert(0, _NumpyCoreCompatFinder())


_ORIGINAL_READ_PICKLE = pd.read_pickle


def read_pickle(path, *args, **kwargs) -> pd.DataFrame:
    """Load a pickle, including files written with NumPy 2.x on NumPy 1.x."""
    patch_numpy2_unpickle_compat()
    return _ORIGINAL_READ_PICKLE(path, *args, **kwargs)


def _read_pickle_with_numpy_compat(path, *args, **kwargs):
    patch_numpy2_unpickle_compat()
    return _ORIGINAL_READ_PICKLE(path, *args, **kwargs)


pd.read_pickle = _read_pickle_with_numpy_compat

# Pickles in this project were often saved with NumPy 2.x; keep NumPy 1.x for JAX.
patch_numpy2_unpickle_compat()

def split_circulating_and_future(
    df: pd.DataFrame,
    cutoff_date_str: str = "2020-04-01",
    left_buffer_days: int = 4,
    right_buffer_days: int = 4,
    circulation_months: int = None,
    circulation_days: int = 15,
    future_months: int = 12,
    future_days: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into circulating and future segments around a cutoff date.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing a 'Date' column.
    cutoff_date_str : str, default="2020-04-01"
        Cutoff date (YYYY-MM-DD) used to segment data.
    left_buffer_days : int, default=4
        Days before cutoff to exclude from circulating data.
    right_buffer_days : int, default=4
        Days after cutoff to exclude from future data.
    circulation_months : int or None, default=None
        Number of months before cutoff to include in circulating data.
        If None, no month offset is applied.
    circulation_days : int, default=15
        Additional days before cutoff to include in circulating data.
    future_months : int, default=12
        Number of months after cutoff to include in future data.
    future_days : int or None, default=None
        If provided, overrides `future_months` to set future data range in days.

    Returns
    -------
    current_df : pd.DataFrame
        Dataframe segment representing circulating period before cutoff.
    future_df : pd.DataFrame
        Dataframe segment representing future period after cutoff.
    """
    # Ensure the Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])

    # Convert version date to timestamp
    t = pd.Timestamp(cutoff_date_str)

    # Calculate buffers
    t_minus_buffer = t - pd.Timedelta(days=left_buffer_days)
    t_plus_buffer = t + pd.Timedelta(days=right_buffer_days)

    # Calculate lower bound for circulating period
    t_minus = t_minus_buffer
    if circulation_months is not None:
        t_minus -= pd.DateOffset(months=circulation_months)
    if circulation_days is not None:
        t_minus -= pd.Timedelta(days=circulation_days)

    # Filter circulating segment
    current_df = df[(df["Date"] >= t_minus) & (df["Date"] < t_minus_buffer)]

    # Calculate upper bound for future period
    if future_days is not None:
        t_plus_future = t_plus_buffer + pd.Timedelta(days=future_days)
    else:
        t_plus_future = t_plus_buffer + pd.DateOffset(months=future_months)

    # Filter future segment
    future_df = df[(df["Date"] >= t_plus_buffer) & (df["Date"] < t_plus_future)]

    return current_df, future_df


def slice_by_cumulative_windows(
    future_df: pd.DataFrame,
    cumulative_windows: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Slice the future dataframe into cumulative time windows by month length.

    Parameters
    ----------
    future_df : pd.DataFrame
        Dataframe containing a 'Date' column representing the future period.
    cumulative_windows : list of str
        List of month-window strings (e.g. ['3M', '6M', '9M', '12M']).

    Returns
    -------
    dict of {str: pd.DataFrame}
        Dictionary mapping each cumulative window string to the corresponding sliced dataframe.
    """
    # Ensure the Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(future_df["Date"]):
        future_df = future_df.copy()
        future_df["Date"] = pd.to_datetime(future_df["Date"])

    # Get the start date of future_df (minimum date)
    start_date = future_df["Date"].min()

    slices_dict = {}
    for window_str in cumulative_windows:
        # Extract the number of months as integer (e.g. '3M' -> 3)
        months = int(window_str.rstrip('M'))

        # Calculate end date for this cumulative window
        end_date = start_date + pd.DateOffset(months=months)

        # Slice the dataframe between start_date and end_date (cumulative)
        sliced_df = future_df[(future_df["Date"] >= start_date) & 
                              (future_df["Date"] < end_date)]
        slices_dict[window_str] = sliced_df

    return slices_dict

def slice_by_disjoint_windows(
    future_df: pd.DataFrame,
    disjoint_windows: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Slice the future dataframe into disjoint (non-overlapping) time windows by month length.

    Parameters
    ----------
    future_df : pd.DataFrame
        Dataframe containing a 'Date' column representing the future period.
    disjoint_windows : list of str
        List of disjoint window strings in the format '<duration>_<block_num>', e.g., ['3M_1', '3M_2'].

    Returns
    -------
    dict of {str: pd.DataFrame}
        Dictionary mapping each disjoint window string to the corresponding sliced dataframe.
    """
    # Ensure the Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(future_df["Date"]):
        future_df = future_df.copy()
        future_df["Date"] = pd.to_datetime(future_df["Date"])

    start_date = future_df["Date"].min()
    slices_dict = {}

    # Extract base duration from the first window (e.g. '3M_1' -> '3M')
    base_duration = disjoint_windows[0].split('_')[0]
    months_per_block = int(base_duration.rstrip('M'))

    for window_str in disjoint_windows:
        # Extract block number (1-indexed)
        block_num = int(window_str.split('_')[1])

        # Compute start and end for this block
        block_start = start_date + pd.DateOffset(months=months_per_block * (block_num - 1))
        block_end = block_start + pd.DateOffset(months=months_per_block)

        # Slice the DataFrame for this block
        sliced_df = future_df[(future_df["Date"] >= block_start) & (future_df["Date"] < block_end)]
        slices_dict[window_str] = sliced_df

    return slices_dict

def compute_summary_stats(
    current_df: pd.DataFrame,
    future_dfs: Dict[str, pd.DataFrame],
    cutoff_key: str
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Compute spike sequence and tip count stats for current and future datasets.

    Args:
        current_df (pd.DataFrame): DataFrame with Sequence_ID and Tip_Count for current period.
        future_dfs (Dict[str, pd.DataFrame]): Dictionary of future DataFrames keyed by period (e.g., '3M').
        cutoff_key (str): A string key for storing this group of stats (e.g., the cutoff date).

    Returns:
        Dict[str, Dict[str, pd.DataFrame]]: Nested dictionary with stats stored per cutoff and period.
    """
    
    stats = {cutoff_key: {}}

    # --- Current data stats ---
    unique_current_seqs = current_df['Sequence_ID'].nunique()
    total_current_tips = current_df['Tip_Count'].sum()
    stats[cutoff_key]['Current'] = pd.DataFrame([{
        "Unique Spike Sequences": unique_current_seqs,
        "Total Tip Count": total_current_tips
    }])

    # --- Track current Sequence_IDs ---
    current_seq_ids = set(current_df['Sequence_ID'])

    # --- Loop through each future window ---
    for period, future_df in future_dfs.items():
        unique_future_seqs = future_df['Sequence_ID'].nunique()
        total_future_tips = future_df['Tip_Count'].sum()

        future_seq_ids = set(future_df['Sequence_ID'])
        common_seq_ids = current_seq_ids & future_seq_ids
        num_common_seqs = len(common_seq_ids)

        common_tip_count = (
            future_df[future_df['Sequence_ID'].isin(common_seq_ids)]['Tip_Count'].sum()
        )

        pct_common_tips = (
            100 * common_tip_count / total_future_tips
            if total_future_tips > 0 else 0
        )

        stats[cutoff_key][period] = pd.DataFrame([{
            "Unique Spike Sequences": unique_future_seqs,
            "Total Tip Count": total_future_tips,
            "Number of Seen Spikes": num_common_seqs,
            "% Tip Count of Seen Spikes": round(pct_common_tips, 2)
        }])

    return stats

def pretty_print_stats(summary_stats: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """
    Nicely prints the nested summary stats dictionary.

    Args:
        summary_stats (Dict[str, Dict[str, pd.DataFrame]]): Nested stats with [cutoff][period] -> DataFrame
    """
    for cutoff, period_dict in summary_stats.items():
        print(f"\n=== Cutoff: {cutoff} ===")
        for period, df in period_dict.items():
            print(f"\n-- {period} --")
            # display(df)  # Use display in notebooks, or print(df.to_string(index=False)) in scripts


def getReference(reference_path: str):
    """
    Load the reference sequence from a FASTA file.

    Args:
        reference_path (str): Path to the reference FASTA file.

    Returns:
        Bio.Seq.Seq: The sequence object from the first record in the FASTA file.
    """
    ref = SeqIO.parse(reference_path, "fasta")
    for record in ref:
        return record.seq


def get_mutation_pattern(mutation_string: str):
    """
    Parse a mutation string into old amino acid, site index, and new amino acid.

    Mutation strings are expected to be in the form 'S:X###Y', where:
        - X = original amino acid (single letter)
        - ### = site position (1-indexed)
        - Y = new amino acid (single letter)
        - '*' indicates a stop codon; if found, returns (0, 0, 0).

    Args:
        mutation_string (str): Mutation in the form 'S:X###Y'.

    Returns:
        Tuple[Union[str, int], Union[int, int], Union[str, int]]:
            (old_aa, site, new_aa) if valid; otherwise (0, 0, 0).
    """
    import re
    pattern = r"S:([A-Za-z*])(\d+)([A-Za-z*])"
    match = re.match(pattern, mutation_string)
    if match:
        old_aa = match.group(1)
        site = int(match.group(2))
        new_aa = match.group(3)
        if old_aa == '*' or new_aa == '*':
            return 0, 0, 0
        return old_aa, site, new_aa
    return 0, 0, 0


def apply_mutations_flat_list(mutation_list: list, refSeq: str):
    """
    Apply a list of amino acid mutations to a reference sequence.

    Uses `get_mutation_pattern` to parse each mutation string and 
    replaces the corresponding position in the sequence.

    Args:
        mutation_list (list): List of mutation strings in the form 'S:X###Y'.
        refSeq (str): The reference amino acid sequence.

    Returns:
        str: The mutated sequence as a string.
    """
    seq = list(refSeq)
    for mut in mutation_list:
        old, site, new = get_mutation_pattern(mut)
        if site == 0:
            continue
        idx = site - 1
        if 0 <= idx < len(seq):
            seq[idx] = new
    return ''.join(seq)


def compute_mutations(seq: str, ref_seq: str) -> tuple:
    """
    Compute amino acid mutations between a sequence and a reference sequence.

    Each differing position is represented in the form 'S:<oldAA><site><newAA>'.

    Args:
        seq (str): The amino acid sequence to compare.
        ref_seq (str): The reference amino acid sequence.

    Returns:
        tuple: A tuple of mutation strings, one for each site that differs.
    """
    mutations = []
    for i, (ref_aa, seq_aa) in enumerate(zip(ref_seq, seq)):
        if ref_aa != seq_aa:
            site = i + 1  # convert zero-index to one-index
            mutations.append(f"S:{ref_aa}{site}{seq_aa}")
    return tuple(mutations)


def add_mutations_column(df, id_to_seq: dict, ref_seq: str):
    """
    Add a 'Mutations' column to a DataFrame of sequences.

    Uses `compute_mutations` to derive mutations relative to the reference
    sequence. Sequences not found in `id_to_seq` will receive an empty tuple.

    Args:
        df (pd.DataFrame): DataFrame containing a 'Sequence_ID' column.
        id_to_seq (dict): Mapping from Sequence_ID to amino acid sequence.
        ref_seq (str): Reference amino acid sequence.

    Returns:
        pd.DataFrame: Copy of input DataFrame with an added 'Mutations' column.
    """
    mutations_list = []
    for seq_id in df["Sequence_ID"]:
        seq = id_to_seq.get(seq_id)
        if seq is None:
            mutations_list.append(())
        else:
            mutations_list.append(compute_mutations(seq, ref_seq))

    df = df.copy()
    df["Mutations"] = mutations_list
    return df

def compute_fitness_score_pyro(mutations: tuple, score_dict: dict) -> float:
    """
    Compute the total fitness score of a sequence using the PyR₀ model.

    Fitness is computed as the product of per-mutation scores.

    Args:
        mutations (tuple): Tuple of mutation strings in the form 'S:X###Y'.
        score_dict (dict): Mapping from mutation string to fitness score.

    Returns:
        float: The total fitness score for the sequence.
    """
    score = 1.0
    for mut in mutations:
        score *= score_dict.get(mut, 1.0)
    return score


def add_fitness_scores_pyro(df, score_dict: dict):
    """
    Add PyR₀ fitness scores to a DataFrame.

    Uses `compute_fitness_score_pyro` to calculate the product of per-mutation
    fitness scores for each sequence.

    Args:
        df (pd.DataFrame): DataFrame containing a 'Mutations' column.
        score_dict (dict): Mapping from mutation string to fitness score.

    Returns:
        pd.DataFrame: Copy of input DataFrame with an added 'Fitness_Score_pyro' column.
    """
    fitness_scores = [compute_fitness_score_pyro(muts, score_dict) for muts in df["Mutations"]]

    df = df.copy()
    df["Fitness_Score_pyro"] = fitness_scores
    return df


def compute_fitness_score_eve(mutations: tuple, score_dict: dict) -> float:
    """
    Compute the total fitness score of a sequence using the EVE model.

    Fitness is computed as the sum of per-mutation scores.
    The mutation prefix 'S:' is removed before lookup.

    Args:
        mutations (tuple): Tuple of mutation strings in the form 'S:X###Y'.
        score_dict (dict): Mapping from mutation site+AA string (e.g. '###Y') to fitness score.

    Returns:
        float: The total fitness score for the sequence.
    """
    score = 0.0
    for mut in mutations:
        score += score_dict.get(mut[2:], 0.0)  # remove 'S:' prefix
    return score


def add_fitness_scores_eve(df, score_dict: dict, mutations_col: str = "Mutations"):
    """
    Add EVE fitness scores to a DataFrame.

    Uses `compute_fitness_score_eve` to calculate the sum of per-mutation
    fitness scores for each sequence.

    Args:
        df (pd.DataFrame): DataFrame containing the mutations column.
        score_dict (dict): Mapping from mutation site+AA string (e.g. '###Y') to fitness score.
        mutations_col (str): Column holding the mutation tuples. Use this when the
            mutations have been re-expressed against a non-Wuhan backbone.

    Returns:
        pd.DataFrame: Copy of input DataFrame with an added 'Fitness_Score_eve' column.
    """
    fitness_scores = [compute_fitness_score_eve(muts, score_dict) for muts in df[mutations_col]]

    df = df.copy()
    df["Fitness_Score_eve"] = fitness_scores
    return df


def build_oracle_current_rankings(
    current_seq_df: pd.DataFrame,
    future_seq_dfs: dict,
    random_state: int = None,
) -> dict:
    """
    Build one oracle ranking of the current sequences per future window.

    For each future slice:
    1. Sort future sequences by ``Tip_Count`` descending.
    2. Keep those present in the current pool (deduplicated) — this order is the
       oracle ranking of current sequences (highest future tips first).
    3. Append the remaining current sequences (absent from this future slice) in
       random order at the end, so the full current pool is ranked for plotting.

    Args:
        current_seq_df (pd.DataFrame): Current sequences with a 'Sequence_ID' column.
        future_seq_dfs (dict): Mapping time_key -> future DataFrame with 'Sequence_ID'
                               and 'Tip_Count' columns.
        random_state (int, optional): Seed for shuffling the appended remainder.

    Returns:
        dict: Mapping time_key -> current_seq_df copy reordered into oracle rank,
              with an added 'Future_Tip_Count' column.
    """
    rng = np.random.default_rng(random_state)
    current_ids = set(current_seq_df["Sequence_ID"])
    oracle_rankings = {}

    for time_key, future_df in future_seq_dfs.items():
        future_sorted = future_df.sort_values("Tip_Count", ascending=False)
        oracle_ids = (
            future_sorted.loc[
                future_sorted["Sequence_ID"].isin(current_ids), "Sequence_ID"
            ]
            .drop_duplicates()
            .tolist()
        )
        oracle_id_set = set(oracle_ids)

        remaining_ids = [
            seq_id
            for seq_id in current_seq_df["Sequence_ID"]
            if seq_id not in oracle_id_set
        ]
        rng.shuffle(remaining_ids)
        ordered_ids = oracle_ids + remaining_ids

        ranked = (
            current_seq_df.set_index("Sequence_ID")
            .loc[ordered_ids]
            .reset_index()
        )
        future_tip_map = future_df.set_index("Sequence_ID")["Tip_Count"]
        ranked["Future_Tip_Count"] = ranked["Sequence_ID"].map(future_tip_map).fillna(0)
        oracle_rankings[time_key] = ranked

    return oracle_rankings


def compute_cumulative_tip_percentages_fast(
    current_seq_df: pd.DataFrame,
    future_seq_dfs: dict,
    oracle_rankings: dict = None,
    max_rank: int = None
) -> dict:
    """
    Compute cumulative percentages of future tips captured by top-ranked sequences.

    For each ranking method (PyR₀, EVEscape, CoVFit, Tip Count, Oracle), this function
    calculates the cumulative fraction of future sequence tips captured when selecting
    the top sequences ranked by fitness, abundance, or oracle future-tip order among
    sequences present in the current pool.

    Args:
        current_seq_df (pd.DataFrame): DataFrame of current sequences with fitness scores.
        future_seq_dfs (dict): Dictionary mapping time keys to DataFrames of future sequences
                               with 'Sequence_ID' and 'Tip_Count' columns.
        oracle_rankings (dict, optional): Mapping time_key -> oracle-ranked current_seq_df
                               (from ``build_oracle_current_rankings``). When provided, adds
                               an 'oracle' ranking method.
        max_rank (int, optional): Maximum number of top sequences to consider. Defaults to all.

    Returns:
        dict: Dictionary mapping time keys to DataFrames containing cumulative tip percentages
              for each ranking method.
    """
    result_dfs = {}

    for time_key, future_df in future_seq_dfs.items():
        current_full = current_seq_df.copy()
        future_tip_map = future_df.set_index("Sequence_ID")["Tip_Count"].to_dict()
        total_future_tips = future_df["Tip_Count"].sum()

        max_rank_to_use = len(current_full) if max_rank is None else min(len(current_full), max_rank)
        results = {"Rank": list(range(1, max_rank_to_use + 1))}

        # --- Define ranking methods ---
        rank_methods = {
            "pyro": current_full.sort_values("Fitness_Score_pyro", ascending=False),
            "eve": current_full.sort_values("Fitness_Score_eve", ascending=False),
            "covfit": current_full.sort_values("fitness_mean", ascending=False),
            "tip_count": current_full.sort_values("Tip_Count", ascending=False),
        }
        if oracle_rankings is not None and time_key in oracle_rankings:
            rank_methods["oracle"] = oracle_rankings[time_key]

        # --- Compute cumulative percentages ---
        for method_key, ranked_df in rank_methods.items():
            top_ids = ranked_df["Sequence_ID"].iloc[:max_rank_to_use].values
            future_tips = np.array([future_tip_map.get(seq_id, 0) for seq_id in top_ids], dtype=np.float32)
            cumulative = np.cumsum(future_tips)
            pct_cumulative = cumulative / total_future_tips if total_future_tips > 0 else np.zeros_like(cumulative)
            results[f"%_tips_{method_key}"] = pct_cumulative

        result_dfs[time_key] = pd.DataFrame(results)

    return result_dfs


def compute_auc_for_rank_results(rank_results: dict) -> dict:
    """
    Compute normalized AUC for cumulative tip percentages across ranks.

    For each time window and ranking method, this function calculates the area under
    the cumulative tip percentage curve and normalizes it by the maximum rank (0–1 scale).

    Args:
        rank_results (dict): Dictionary mapping time keys to DataFrames containing
                             cumulative tip percentages for each ranking method.

    Returns:
        dict: Nested dictionary mapping time_key -> method -> normalized AUC.
    """
    auc_dict = {}

    for time_key, df in rank_results.items():
        max_rank = df["Rank"].max()
        tip_cols = [col for col in df.columns if col.startswith("%_tips_")]
        auc_dict[time_key] = {}

        for col in tip_cols:
            method = col.replace("%_tips_", "")
            auc = np.trapz(df[col], df["Rank"])
            # The curve is already a fraction of the window's total future tips,
            # so the reference is the rectangle of height 1 spanning the rank
            # axis -- every future sequence captured at every rank. Its area is
            # max_rank, which puts the result on a 0-1 scale.
            auc_normalized = auc / max_rank
            auc_dict[time_key][method] = auc_normalized

    return auc_dict

def build_node_path(sample_id, parent_lookup):
    path = []
    current_node = sample_id
    # Keep moving up the tree until no parent is found or root (assuming root has None or missing parent)
    while current_node not in (None, 'NULL'):
        path.append(current_node)
        if current_node not in parent_lookup:
            print(f"\nMissing '{current_node}' in database\n")
            break
        current_node = parent_lookup[current_node]
        
    # Add the root node (which has no parent)
    path.append(current_node)
    # Reverse to get path from root -> sample_id
    path.reverse()
    return path

def find_anchor_remainder_reverse(node_path, nodes_before_cutoff):
    # Traverse backwards from the end
    for i in range(len(node_path) - 1, -1, -1):
        node = node_path[i]
        if node in nodes_before_cutoff:
            anchor = node
            remainder = node_path[i+1:]  # nodes after anchor towards the sample
            return (anchor, remainder)
    # If no anchor found
    return (None, node_path)

def get_mutations_from_remainder(anchor_remainder, mutations_lookup):
    """
    Collect spike mutation strings along the remainder path (nodes after the anchor).
    Nodes are walked from tip toward anchor (``reversed`` remainder order) so mutations
    nearer the leaf win when the same site appears multiple times.

    Lookup values that are a single string are treated as one field; ``;``-separated
    tokens are split into separate entries. Per-site deduplication: after a site is
    represented once (via ``get_mutation_pattern``), later tokens for that site are
    skipped. Tokens that do not match the ``S:X###Y`` pattern dedupe by full string.
    """
    remainder_nodes = anchor_remainder[1]
    mutations = []
    covered_keys = set()
    for node in reversed(remainder_nodes):
        muts = mutations_lookup.get(node, [])
        if isinstance(muts, str):
            muts = [muts]
        elif muts is None:
            muts = []
        for mut in muts:
            if not mut:
                continue
            for piece in str(mut).split(";"):
                piece = piece.strip()
                if not piece:
                    continue
                _, site, _ = get_mutation_pattern(piece)
                key = site if site != 0 else piece
                if key in covered_keys:
                    continue
                covered_keys.add(key)
                mutations.append(piece)
    return mutations