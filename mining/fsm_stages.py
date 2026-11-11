"""Miner FSM stages S1–S5 (gap → evidence → synthesis → persist → report)."""
from __future__ import annotations

from typing import Optional, Set

# Allowed forward edges (and self-loops for multi-round CE / evidence).
ALLOWED = {
    "S1": {"S1", "S2"},
    "S2": {"S2", "S3"},
    "S3": {"S3", "S4", "S2"},  # S2: need more evidence after failed CE
    "S4": {"S4", "S5"},
    "S5": {"S5", "S1"},  # next mining iteration
}


class FSMError(Exception):
    pass


class MiningFSM:
    def __init__(self, stage: str = "S1"):
        self.stage = stage
        self.history = [stage]

    def advance(self, nxt: str, *, force: bool = False) -> str:
        if force:
            self.stage = nxt
            self.history.append(nxt)
            return nxt
        allowed: Set[str] = ALLOWED.get(self.stage, set())
        if nxt not in allowed:
            raise FSMError(f"illegal transition {self.stage}->{nxt}; allowed={sorted(allowed)}")
        self.stage = nxt
        self.history.append(nxt)
        print(f"[trainaudit] FSM {self.history[-2] if len(self.history)>1 else '?'}->{nxt}")
        return nxt

    def ensure(self, required: str) -> Optional[str]:
        """Return error string if current stage is before required (order S1<S2<S3<S4<S5)."""
        order = ["S1", "S2", "S3", "S4", "S5"]
        if self.stage not in order or required not in order:
            return f"unknown stage cur={self.stage} need={required}"
        if order.index(self.stage) < order.index(required):
            return f"need stage>={required}, current={self.stage}"
        return None
