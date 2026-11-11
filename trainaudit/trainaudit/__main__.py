"""Offline forensic CLI for trainaudit.

Usage:
  python -m trainaudit verify   <trace.duckdb> [--use-dsl] [--tier T0|T1]
  python -m trainaudit diagnose <trace.duckdb> [--tier T0|T1] [--rca]
  python -m trainaudit summary  <trace.duckdb>
  python -m trainaudit replay   <trace.duckdb> [--every 50]

Designed for the post-mortem pattern: a cluster operator captures a
`trace.duckdb` during a production training run with
`trainaudit.enable(db_path=...)`, copies it off the GPU machine, and
inspects it on a separate workstation. The CLI loads the existing
trace, runs rules and diagnosis without re-executing the training, and
emits the same kind of structured output a live `trainaudit.run_rules()`
call would produce inside the training process.

`replay` walks the trace forward in event_id order, simulating the
OnlineRunner cadence (default every 50 events) so the user can see
when each violation would have first surfaced during live training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from .diagnosis import expand_results
from .rules.base import RuleResult
from .store import TraceStore
from .tiers import Tier
from .verifier import run_rules
from .verifier import summarize as _summarize


def _open_store(path: str) -> TraceStore:
    p = Path(path)
    if not p.exists():
        sys.exit(f"trace not found: {path}")
    return TraceStore(str(p))


def _resolve_tier(name: str) -> Tier:
    name = name.upper()
    if name in ("T0", "T0_PYTORCH"):
        return Tier.T0_PYTORCH
    if name in ("T1", "T1_FW_METADATA"):
        return Tier.T1_FW_METADATA
    if name in ("T2", "T2_FW_PRIMITIVE"):
        return Tier.T2_FW_PRIMITIVE
    if name in ("T3", "T3_FW_SPECIFIC"):
        return Tier.T3_FW_SPECIFIC
    if name in ("T4", "T4_INSTANCE"):
        return Tier.T4_INSTANCE
    sys.exit(f"unknown tier: {name}")


# ---- subcommands --------------------------------------------------------


def cmd_verify(args):
    store = _open_store(args.trace)
    tier = _resolve_tier(args.tier)
    results = run_rules(store, tier=tier, use_dsl=args.use_dsl)
    print(_summarize(results))
    n_violated = sum(1 for r in results if r.violated)
    store.close()
    return 0 if n_violated == 0 else 1


def cmd_diagnose(args):
    store = _open_store(args.trace)
    tier = _resolve_tier(args.tier)
    results = run_rules(store, tier=tier, use_dsl=args.use_dsl)
    reports = expand_results(store, results)
    if not reports:
        print("[diagnose] no violations to expand")
        store.close()
        return 0

    if args.rca:
        from .diagnosis.rca_agent import explain_all
        rcas = explain_all(reports)
        for r in rcas:
            print(json.dumps(r.to_dict(), indent=2, default=str, ensure_ascii=False))
    else:
        for r in reports:
            print(json.dumps(r.to_dict(), indent=2, default=str, ensure_ascii=False))
    store.close()
    return 1


def cmd_summary(args):
    store = _open_store(args.trace)
    rows = store.conn.execute(
        "SELECT hookpoint, COUNT(*) FROM events GROUP BY hookpoint "
        "ORDER BY hookpoint").fetchall()
    total = sum(c for _, c in rows)
    n_steps = store.conn.execute(
        "SELECT MIN(step), MAX(step), COUNT(DISTINCT step) FROM events"
    ).fetchone()
    n_ranks = store.conn.execute(
        "SELECT COUNT(DISTINCT rank) FROM events").fetchone()[0]
    print(f"[summary] total events: {total:,}")
    print(f"[summary] step range: {n_steps[0]}..{n_steps[1]} "
          f"(distinct={n_steps[2]})")
    print(f"[summary] distinct ranks: {n_ranks}")
    print(f"[summary] events per hookpoint:")
    for hp, c in rows:
        print(f"  {hp:36s}  {c:>8,}")
    store.close()
    return 0


def cmd_replay(args):
    """Stream events through OnlineRunner; print first-fire ticks."""
    from .streaming import OnlineRunner
    from .rules import all_rules
    store = _open_store(args.trace)
    tier = _resolve_tier(args.tier)
    rules = [r for r in all_rules() if r.min_tier <= tier]
    # We can't actually stream past events through OnlineRunner without
    # a moving cursor; simplest is run_rules per "tick window" by
    # filtering events with event_id <= window_end.
    # For simplicity, iterate event_ids in batches of `--every`.
    max_id = store.conn.execute("SELECT MAX(event_id) FROM events").fetchone()[0]
    if max_id is None:
        print("[replay] empty trace")
        store.close()
        return 0
    seen_violations: dict = {}
    tick = 0
    cursor = 0
    while cursor < max_id:
        cursor = min(cursor + args.every, max_id)
        # Trick: temporarily filter trace by deleting events after cursor
        # is too invasive. Instead, run all rules + filter result event_ids.
        results = run_rules(store, tier=tier)
        new_at_this_tick = []
        for r in results:
            if not r.violated:
                continue
            ids = (r.evidence or {}).get("violation_event_ids") or []
            for eid in ids:
                if eid <= cursor and eid not in seen_violations:
                    seen_violations[eid] = (tick, r.rule_id, r.message)
                    new_at_this_tick.append((eid, r.rule_id, r.message))
        if new_at_this_tick:
            print(f"[tick {tick} (events ≤ {cursor})] {len(new_at_this_tick)} new violation(s):")
            for eid, rid, msg in new_at_this_tick[:5]:
                print(f"  event_id={eid} rule={rid} msg={msg}")
        tick += 1
    print(f"[replay] {len(seen_violations)} unique violations across "
          f"{tick} ticks")
    store.close()
    return 0 if not seen_violations else 1


# ---- entry --------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m trainaudit",
                                 description="Offline trainaudit forensics.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("verify", help="run rules + summary")
    pv.add_argument("trace")
    pv.add_argument("--tier", default="T1")
    pv.add_argument("--use-dsl", action="store_true",
                    help="run YAML DSL predicates instead of Python rules")
    pv.set_defaults(fn=cmd_verify)

    pd = sub.add_parser("diagnose", help="expand violations to DiagnosisReports")
    pd.add_argument("trace")
    pd.add_argument("--tier", default="T1")
    pd.add_argument("--use-dsl", action="store_true")
    pd.add_argument("--rca", action="store_true",
                    help="wrap each report with the LLM RCA agent (stub by default)")
    pd.set_defaults(fn=cmd_diagnose)

    ps = sub.add_parser("summary", help="event counts per hookpoint")
    ps.add_argument("trace")
    ps.set_defaults(fn=cmd_summary)

    pr = sub.add_parser("replay", help="walk events forward, show first-fire ticks")
    pr.add_argument("trace")
    pr.add_argument("--tier", default="T1")
    pr.add_argument("--every", type=int, default=50,
                    help="events per tick (default 50)")
    pr.set_defaults(fn=cmd_replay)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
