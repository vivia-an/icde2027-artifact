"""DuckDB-backed append-only trace store.

Two execution modes:

  - **sync** (default): `emit()` does json.dumps + appends to in-memory
    buffer; `_flush_locked()` does executemany at buffer_limit. Predictable
    and ordering-deterministic. Used by tests and CPU smoke runs.

  - **async** (opt-in via `async_mode=True`): `emit()` increments the
    counter and pushes the raw payload (un-serialised) onto a thread-safe
    queue, returning the event_id immediately. A background daemon thread
    drains the queue, serialises payloads, and writes to DuckDB in batches.
    This pushes the json.dumps + INSERT cost off the training step's
    critical path. Used in production to hit paper §4.3 <5% overhead.

The two modes are wire-compatible: same events table schema, same event_id
ordering, same `flush()` semantics for reads. Production code calls
`enable(..., async_mode=True)`; rule code can ignore the difference because
`flush()` waits for the queue to drain before any read.

Thread-safety contract (engine fix 2026-10-25, see
benchmark/eval/icde_v1/00_freeze/RULE_CHANGELOG.md, ENGINE-001):

  - `self.conn` (the writer connection) is touched only while holding
    `self._lock`. In async mode the drain thread is its only user.
    A DuckDB connection object is not thread-safe; the pre-fix engine let
    `OnlineRunner.tick()` and `query()/num_events()` run statements on
    `self.conn` concurrently with the drain thread's INSERTs, which
    corrupts the client state (SIGSEGV / heap corruption, or
    "closed pending query result" errors swallowed as rule runtime
    errors).
  - Readers on other threads call `reader()` to get their OWN DuckDB
    connection (`conn.cursor()`) onto the same database instance; DuckDB
    supports one connection per thread concurrently (MVCC).
  - Event-id assignment and async enqueue happen under one lock, so the
    queue is FIFO in event_id order and the rows visible to any reader
    are always the contiguous prefix 1..max(event_id).
  - `flush()` waits until every event enqueued before the call has been
    written (or counted as dropped/failed), not merely until the queue
    looks empty (the pre-fix check raced with the in-flight batch).

Insert path (ENGINE-002): batches are appended as one Arrow table
(`INSERT INTO events SELECT * FROM <arrow>`) instead of `executemany`;
same rows, same order, ~2k -> ~1e5 rows/s measured on the CPU node.
"""
import atexit
import json
import queue
import threading
import time
from typing import Any, Dict, Optional, Tuple

import duckdb

try:  # columnar bulk insert (ENGINE-002); executemany fallback if missing
    import pyarrow as _pa
except Exception:  # noqa: BLE001
    _pa = None

from .schema import SCHEMA_VERSION

# Internal type for an in-flight async event before json.dumps. Carrying
# the raw `payload` dict rather than the serialised string is what shifts
# the cost off the critical path.
_PendingEvent = Tuple[int, int, int, str, int, Dict[str, Any]]


class TraceStore:
    """One DuckDB file, one events table, JSON payload.

    Schema is intentionally minimal. Rules read it via SQL or .query().
    """

    def __init__(self, db_path: str = ":memory:", *,
                 async_mode: bool = False,
                 async_queue_size: int = 8192,
                 async_drain_interval_s: float = 0.2):
        self.db_path = db_path
        self.conn = duckdb.connect(str(db_path))
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id        BIGINT  PRIMARY KEY,
                step            INTEGER,
                rank            INTEGER,
                hookpoint       VARCHAR,
                ts_ns           BIGINT,
                schema_version  INTEGER,
                payload         VARCHAR
            )
        """)
        self._counter = 0
        # _lock guards self.conn + the sync buffer (DB lock).
        # _id_lock guards event-id assignment + async enqueue order.
        self._lock = threading.Lock()
        self._id_lock = threading.Lock()
        self._buffer = []
        self._buffer_limit = 256
        self._step = 0
        self._rank = 0
        self._closed = False

        # Async-mode state. All `_async_*` attributes only meaningful when
        # async_mode=True.
        self._async_mode = async_mode
        self._async_q: Optional[queue.Queue] = None
        self._async_thread: Optional[threading.Thread] = None
        self._async_done: Optional[threading.Event] = None
        self._async_drain_interval = async_drain_interval_s
        # health counters — readable via `health()`
        self._async_dropped = 0
        self._async_serialize_failures = 0
        self._async_db_failures = 0
        # enqueued vs. accounted (written + failed) counters; flush() waits
        # on _async_cv until accounted >= the enqueued count it observed.
        self._async_enqueued = 0
        self._async_accounted = 0
        self._async_cv = threading.Condition()
        if async_mode:
            self._async_q = queue.Queue(maxsize=async_queue_size)
            self._async_done = threading.Event()
            self._async_thread = threading.Thread(
                target=self._async_drain_loop, daemon=True,
                name="trainaudit-store-drain")
            self._async_thread.start()
            atexit.register(self._async_shutdown)

    # ---- public API ----

    def set_step(self, step: int) -> None:
        self._step = int(step)

    def set_rank(self, rank: int) -> None:
        self._rank = int(rank)

    def emit(self, hookpoint: str, payload: Dict[str, Any], *,
             step: Optional[int] = None, rank: Optional[int] = None) -> int:
        """Append one event. Returns event_id (or -1 if store is closed).

        Sync mode: serialises payload + buffers; flushes at buffer_limit.
        Async mode: pushes raw payload onto queue; returns immediately.
        Counter is monotonically incremented under lock in either mode so
        event_id ordering matches emit-call ordering across threads."""
        if self._closed:
            return -1

        cur_step = self._step if step is None else int(step)
        cur_rank = self._rank if rank is None else int(rank)

        if self._async_mode:
            # id assignment + enqueue under one lock => queue is FIFO in
            # event_id order (readers always see a contiguous id prefix).
            with self._id_lock:
                self._counter += 1
                event_id = self._counter
                pending: _PendingEvent = (event_id, cur_step, cur_rank,
                                           hookpoint, time.time_ns(), payload)
                try:
                    self._async_q.put_nowait(pending)
                    self._async_enqueued += 1
                except queue.Full:
                    # Back-pressure: try briefly blocking before dropping.
                    # Dropping silently corrupts trace, so retry once.
                    try:
                        self._async_q.put(pending, timeout=0.05)
                        self._async_enqueued += 1
                    except queue.Full:
                        self._async_dropped += 1
            return event_id

        with self._id_lock:
            self._counter += 1
            event_id = self._counter
        ts = time.time_ns()

        # Sync path: serialise here.
        with self._lock:
            self._buffer.append((
                event_id, cur_step, cur_rank, hookpoint, ts,
                SCHEMA_VERSION,
                json.dumps(payload, default=_json_default),
            ))
            if len(self._buffer) >= self._buffer_limit:
                self._flush_locked()
        return event_id

    def flush(self) -> None:
        """Wait for all pending events (async queue + sync buffer) to land
        in DuckDB. Safe for rule code to call before reading."""
        if self._async_mode and self._async_q is not None:
            # Wait until every event enqueued before this call is written
            # (best-effort with timeout, as before).
            with self._id_lock:
                target = self._async_enqueued
            deadline = time.monotonic() + 30.0
            with self._async_cv:
                while (self._async_accounted < target and not self._closed
                       and self._async_thread is not None
                       and self._async_thread.is_alive()):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._async_cv.wait(timeout=min(remaining, 0.05))
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        if self._closed:
            return
        if self._async_mode:
            self._async_shutdown()
        with self._lock:
            self._flush_locked()
            self.conn.close()
            self._closed = True

    def num_events(self) -> int:
        self.flush()
        with self._lock:
            return self.conn.execute(
                "SELECT COUNT(*) FROM events").fetchone()[0]

    def query(self, sql: str, *params):
        self.flush()
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def reader(self):
        """Return a NEW DuckDB connection onto the same database instance,
        for exclusive use by one reader thread (e.g. an OnlineRunner on a
        background verifier thread). Never share the returned object
        across threads; never use `self.conn` outside `self._lock`."""
        with self._lock:
            return self.conn.cursor()

    def health(self) -> Dict[str, Any]:
        """Return counters useful for production monitoring.

        Production should poll this every minute and alert when any
        non-zero counter increases — a healthy run keeps all 0."""
        out: Dict[str, Any] = {
            "async_mode": self._async_mode,
            "closed": self._closed,
            "counter": self._counter,
            "buffer_size": len(self._buffer),
        }
        if self._async_mode and self._async_q is not None:
            out.update({
                "queue_size": self._async_q.qsize(),
                "queue_capacity": self._async_q.maxsize,
                "dropped": self._async_dropped,
                "serialize_failures": self._async_serialize_failures,
                "db_failures": self._async_db_failures,
                "thread_alive": self._async_thread.is_alive()
                if self._async_thread else False,
            })
        return out

    # ---- internals ----

    def _insert_rows_locked(self, rows) -> None:
        """Append `rows` (list of 7-tuples, event_id order) to events.
        Caller holds self._lock."""
        if _pa is None:
            self.conn.executemany(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
            return
        eid, st, rk, hp, ts, sv, pl = zip(*rows)
        _batch = _pa.table({
            "event_id": _pa.array(eid, _pa.int64()),
            "step": _pa.array(st, _pa.int32()),
            "rank": _pa.array(rk, _pa.int32()),
            "hookpoint": _pa.array(hp, _pa.string()),
            "ts_ns": _pa.array(ts, _pa.int64()),
            "schema_version": _pa.array(sv, _pa.int32()),
            "payload": _pa.array(pl, _pa.string()),
        })
        self.conn.register("_trainaudit_batch", _batch)
        try:
            self.conn.execute(
                "INSERT INTO events SELECT * FROM _trainaudit_batch")
        finally:
            self.conn.unregister("_trainaudit_batch")

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        try:
            self._insert_rows_locked(self._buffer)
        except Exception:
            # In async mode this can race with concurrent close; surface as
            # a counter rather than crash training.
            if self._async_mode:
                self._async_db_failures += 1
            else:
                raise
        self._buffer.clear()

    def _async_drain_loop(self) -> None:
        """Run in the background thread. Pulls pending events off the
        queue, serialises payloads (json.dumps), batches into _buffer,
        flushes to DuckDB. Exits when `_async_done` is set AND queue is
        empty."""
        local_buf: list = []
        while not (self._async_done.is_set() and self._async_q.empty()):
            try:
                pending = self._async_q.get(timeout=self._async_drain_interval)
            except queue.Empty:
                pending = None
            if pending is not None:
                local_buf.append(self._serialise(pending))
                # Drain more without blocking, up to buffer_limit
                while len(local_buf) < self._buffer_limit:
                    try:
                        more = self._async_q.get_nowait()
                    except queue.Empty:
                        break
                    local_buf.append(self._serialise(more))
            if local_buf:
                n = len(local_buf)
                with self._lock:
                    if not self._closed:
                        try:
                            self._insert_rows_locked(local_buf)
                        except Exception:
                            self._async_db_failures += 1
                local_buf.clear()
                with self._async_cv:
                    self._async_accounted += n
                    self._async_cv.notify_all()

    def _serialise(self, pending: _PendingEvent) -> Tuple:
        eid, step, rank, hp, ts, payload = pending
        try:
            pl_str = json.dumps(payload, default=_json_default)
        except Exception as e:  # noqa: BLE001
            self._async_serialize_failures += 1
            pl_str = json.dumps({"_async_serialize_error": str(e)})
        return (eid, step, rank, hp, ts, SCHEMA_VERSION, pl_str)

    def _async_shutdown(self) -> None:
        if not self._async_mode or self._async_done is None:
            return
        if self._async_done.is_set():
            return  # already shutting down
        self._async_done.set()
        if self._async_thread is not None and self._async_thread.is_alive():
            self._async_thread.join(timeout=5.0)


def _json_default(obj):
    """Best-effort JSON encoder for non-trivial types."""
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)
