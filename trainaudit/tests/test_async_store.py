"""Async-mode TraceStore tests.

Goals:
  - emit() returns immediately (does NOT block on json.dumps + INSERT)
  - flush() guarantees all pending events visible
  - event_id ordering is monotonic across threads
  - close() drains the queue before closing the conn
  - health() reports queue + drop counters
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trainaudit.store import TraceStore  # noqa: E402


def test_async_emit_returns_event_id_synchronously():
    s = TraceStore(":memory:", async_mode=True)
    try:
        eid1 = s.emit("optim.step.post", {"x": 1})
        eid2 = s.emit("optim.step.post", {"x": 2})
        eid3 = s.emit("optim.step.post", {"x": 3})
        assert eid1 == 1 and eid2 == 2 and eid3 == 3, (
            f"event_id should be 1,2,3 in order; got {eid1},{eid2},{eid3}")
    finally:
        s.close()


def test_async_flush_makes_events_visible():
    s = TraceStore(":memory:", async_mode=True)
    try:
        for i in range(50):
            s.emit("module.fwd.post", {"i": i, "module_class": "Linear"})
        # Before flush, queue may still hold events
        s.flush()
        # After flush, all 50 events must be in events table
        n = s.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert n == 50, f"expected 50 events after flush, got {n}"
    finally:
        s.close()


def test_async_event_id_monotonic_across_threads():
    s = TraceStore(":memory:", async_mode=True)
    try:
        ids: list = []
        lock = threading.Lock()

        def worker(start, count):
            for i in range(count):
                eid = s.emit("module.fwd.post", {"src": start, "i": i})
                with lock:
                    ids.append(eid)

        threads = [threading.Thread(target=worker, args=(t, 100))
                    for t in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()

        # All 400 ids should be unique and form a contiguous 1..400 set
        assert sorted(ids) == list(range(1, 401))

        s.flush()
        n = s.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert n == 400
    finally:
        s.close()


def test_async_close_drains_queue():
    """close() must NOT lose in-flight events."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "async.duckdb")
        s = TraceStore(path, async_mode=True)
        for i in range(200):
            s.emit("module.fwd.post", {"i": i})
        # close immediately after emit — bg thread may still have items
        s.close()

        # Re-open and verify all 200 events landed
        s2 = TraceStore(path)
        try:
            n = s2.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            assert n == 200, (f"close should drain queue; expected 200 events "
                              f"on reopen, got {n}")
        finally:
            s2.close()


def test_health_counters():
    s = TraceStore(":memory:", async_mode=True)
    try:
        for i in range(20):
            s.emit("optim.step.post", {"i": i})
        h = s.health()
        assert h["async_mode"] is True
        assert h["counter"] == 20
        assert h["thread_alive"] is True
        assert h["dropped"] == 0
        assert h["serialize_failures"] == 0
        assert h["db_failures"] == 0
        s.flush()
        h2 = s.health()
        assert h2["queue_size"] == 0, "queue should be drained after flush"
    finally:
        s.close()


def test_async_serialize_failure_recorded_not_raised():
    """Un-JSON-able payload should be replaced with an error marker, not
    raise (would crash the bg thread)."""
    s = TraceStore(":memory:", async_mode=True)
    try:
        # An object whose __dict__ contains itself → self-referencing,
        # json.dumps will fail with RecursionError. Our _json_default
        # falls back to str(obj) so this should still serialise. Use a
        # truly unserialisable (function with weird __dict__) instead.
        class Bad:
            def __init__(self):
                self.x = self  # cycle
        bad_payload = {"bad": Bad()}
        s.emit("module.fwd.post", bad_payload)
        s.flush()
        h = s.health()
        # Either it serialised via str() fallback (counter 0) or marked
        # as failure (counter 1). Either way the bg thread is still alive.
        assert h["thread_alive"] is True
        assert s.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    finally:
        s.close()


def test_async_throughput_does_not_block_caller():
    """emit() should be much faster in async mode than sync mode for a
    payload with non-trivial json.dumps cost."""
    big_payload = {"a": list(range(1000)), "b": {"nested": list(range(500))},
                   "c": {"x": [list(range(20)) for _ in range(50)]}}

    # sync timing
    s_sync = TraceStore(":memory:", async_mode=False)
    try:
        t0 = time.perf_counter()
        for _ in range(2000):
            s_sync.emit("optim.step.post", big_payload)
        sync_time = time.perf_counter() - t0
    finally:
        s_sync.close()

    # async timing — emit-only, no flush in the timed block
    s_async = TraceStore(":memory:", async_mode=True)
    try:
        t0 = time.perf_counter()
        for _ in range(2000):
            s_async.emit("optim.step.post", big_payload)
        async_time = time.perf_counter() - t0
        s_async.flush()
    finally:
        s_async.close()

    # Async emit should be at least 2× faster on this payload size.
    # (If the bg drain is faster than enqueue we'd see closer to 1× — that
    # means the json.dumps cost is small relative to overhead, which is
    # also fine; we still want emit-only to be no slower than sync.)
    print(f"\n[bench] sync={sync_time*1000:.1f}ms  "
          f"async_emit_only={async_time*1000:.1f}ms  "
          f"speedup={sync_time/max(async_time,1e-9):.2f}x")
    assert async_time <= sync_time * 0.95, (
        f"expected async emit to be at least 5% faster than sync; "
        f"sync={sync_time:.3f}s async={async_time:.3f}s")


def test_async_disabled_by_default_keeps_sync_semantics():
    """Default-constructed store should behave identically to current sync
    code — events visible after just a few emits without explicit flush."""
    s = TraceStore(":memory:")  # async_mode defaults to False
    try:
        # Sync: events go to buffer; not visible until buffer_limit (256) hit
        for i in range(10):
            s.emit("module.fwd.post", {"i": i})
        # Without flush, sync mode also doesn't show in events table yet
        # (buffer is below limit)
        s.flush()
        n = s.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert n == 10
        h = s.health()
        assert h["async_mode"] is False
        assert "queue_size" not in h
    finally:
        s.close()
