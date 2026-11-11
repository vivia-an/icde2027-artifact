"""Find the outlier rank in a replica-equality violation.

When `T1-replica-cksum-equal` fires, the bug payload includes
`gathered_cksums` (one cksum per rank in the replica group). The "wrong"
rank is the one whose cksum differs from the majority. With a 2-rank
group it's ambiguous (both ranks are "outliers") but with ≥3 the bug
location is usually clear-cut (e.g. M-005: rank 0 inits with default
RNG, rank 1 with TP-aware RNG → rank 0 is the outlier vs the rest).
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional


def find_outlier_rank(gathered_cksums: List[Any]) -> Optional[int]:
    """Return the index (rank) whose value differs from the majority,
    or None if there's no clear majority (e.g. 2-rank tie or all same).
    """
    if not gathered_cksums or len(gathered_cksums) < 2:
        return None
    counts = Counter(gathered_cksums)
    if len(counts) == 1:
        return None  # all equal — caller shouldn't have called us
    if len(counts) == 2 and len(gathered_cksums) == 2:
        return None  # 2-rank disagreement; both are equally suspect
    # Pick the value with strict minority count (must be unique)
    sorted_vals = counts.most_common()
    majority_val, majority_count = sorted_vals[0]
    minority_val, minority_count = sorted_vals[-1]
    if minority_count == majority_count:
        return None  # tie
    # First rank that holds the minority value
    for rank, v in enumerate(gathered_cksums):
        if v == minority_val:
            return rank
    return None


def summarize_replica_violation(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a bug_specific dict for T1-replica-cksum-equal violations."""
    out: Dict[str, Any] = {
        "param_name": entry.get("name"),
        "group_size": entry.get("group_size"),
        "gathered_cksums": entry.get("gathered_cksums"),
    }
    gathered = entry.get("gathered_cksums") or []
    out["outlier_rank"] = find_outlier_rank(gathered)
    return out
