"""CPU parallelism helpers (Phase 2c).

The pipeline's independent units (per-array extraction, ...) are embarrassingly parallel.  This
module runs them across processes while preserving the project's hard constraints:

* **On by default, above one item.**  ``jobs`` defaults to the ``PFLM_JOBS`` env var, which itself
  defaults to *all cores*.  ``pmap_shared`` spawns whenever there is more than one work item;
  measured worst-case pool startup is ~2.4 s (16 workers, warm cache), well under the point where it
  could hurt an already-short run, so no size heuristic is needed.  ``PFLM_JOBS=1`` forces serial.
* **Order preserving.**  Results are always returned in input order, so any downstream ``max`` /
  ``sort`` / dedup / CSV ordering resolves ties exactly as it did serially.
* **Deterministic, serial == parallel.**  Both the serial and the worker paths run each task with
  BLAS/OpenMP pinned to one thread, so a thread-count-dependent reduction (e.g. ``np.linalg.lstsq``
  in ``extract._level_floor``) can't shift its last bits between paths or across machines.  The
  per-pixel ``cKDTree`` query is threaded in the single-process path but single-threaded inside
  workers (via ``PFLM_WORKER``) to avoid ``n_workers x n_cores`` oversubscription; its returned
  distances are identical either way (scipy partitions query points, not the reduction).
* **Cheap sharing.**  A large read-only object (the assembled scan) is sent to each worker ONCE via
  the pool initializer, not re-pickled per task.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor

_SHARED = None          # per-worker handle to the large read-only object (set by the initializer)


def resolve_jobs(jobs=None) -> int:
    """Resolve the worker count.

    Precedence: an explicit ``jobs`` argument, else the ``PFLM_JOBS`` env var, else **all cores**
    (parallelism is on by default).  An integer ``<=0`` also means all cores; unparseable junk falls
    back to serial (1).
    """
    if jobs is None:
        jobs = os.environ.get("PFLM_JOBS")     # None when unset -> default-on (all cores) below
    if jobs is None:
        return os.cpu_count() or 1
    try:
        n = int(jobs)
    except (TypeError, ValueError):
        return 1                               # junk env/arg -> conservative serial
    if n <= 0:
        n = os.cpu_count() or 1
    return max(1, n)


@contextmanager
def _single_threaded_blas():
    """Pin BLAS/OpenMP to one thread for the duration of one task.

    Applied identically in the serial and worker paths so a BLAS reduction (``lstsq``) is bit-stable
    between them and reproducible across machines.  No-op if threadpoolctl is unavailable.
    """
    try:
        import threadpoolctl
        with threadpoolctl.threadpool_limits(limits=1):
            yield
    except Exception:
        yield


def _init_worker(shared) -> None:
    global _SHARED
    _SHARED = shared
    os.environ["PFLM_WORKER"] = "1"            # -> extract uses a single cKDTree thread per worker


def _call(args):
    fn, item = args
    with _single_threaded_blas():
        return fn(_SHARED, item)


def pmap_shared(fn, items, shared, jobs=None):
    """Ordered map of ``fn(shared, item)`` over ``items``.

    ``fn`` must be a top-level (picklable) function and every ``item`` must be picklable; ``shared``
    is handed to each worker once via the initializer.  Runs serially (byte-identical) when the
    resolved job count is 1 or there is at most one item; otherwise fans out across processes.
    """
    items = list(items)
    n = resolve_jobs(jobs)
    if n <= 1 or len(items) <= 1:
        with _single_threaded_blas():
            return [fn(shared, it) for it in items]
    with ProcessPoolExecutor(max_workers=min(n, len(items)),
                             initializer=_init_worker, initargs=(shared,)) as ex:
        return list(ex.map(_call, [(fn, it) for it in items]))
