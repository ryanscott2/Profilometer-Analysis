"""Optional FFT / normalized-cross-correlation acceleration backend.

Backend selection order (the user's "GPU primary, CPU fallback" directive):

    CuPy (GPU, float32)  ->  pyfftw (CPU, float64)  ->  scipy.fft (CPU, float64)

DATA-QUALITY CONTRACT (this module must never silently degrade a measured result):

* ``_ncc_scipy`` is **byte-identical** to the historical ``register._pattern_ncc`` (same
  ``scipy.signal.fftconvolve`` float64 maths).  It is the deterministic reproducibility reference
  the self-test pins, selected whenever :func:`force_cpu` / :func:`set_force_cpu` is active.
* A ``decisive`` NCC -- one whose peak VALUE is compared across reflection / rotation hypotheses --
  is ALWAYS computed on the CPU float64 path: a float32 GPU tie could otherwise silently mirror or
  mis-rotate a whole sample.  Only pure *localization* NCCs (peak position, re-refined on CPU by
  ``_refine_origin``) may use the GPU.
* cuFFT is not bit-reproducible across driver / hardware, so the GPU path is never the reproducibility
  path and is excluded from the pinned self-test tolerances (it gets a separate agreement check).

Backend is chosen per call via :func:`select_backend`; the process-wide preference is the
``PFLM_ACCEL`` environment variable (``auto`` | ``cpu``/``scipy`` | ``pyfftw`` | ``cupy``/``gpu``).
``auto`` deliberately stays on the deterministic CPU path for now; GPU is opt-in until the
GPU-proposes / CPU-confirms detection wiring lands (Phase 2b).
"""
from __future__ import annotations

import os
import numpy as np

_FORCE_CPU = False
_CUPY = None            # None = unprobed, False = unavailable, module = available


def set_force_cpu(flag: bool) -> None:
    """Pin the deterministic CPU/scipy path (the self-test calls this so runs are reproducible)."""
    global _FORCE_CPU
    _FORCE_CPU = bool(flag)


class force_cpu:
    """Context-manager form of :func:`set_force_cpu`."""

    def __enter__(self):
        global _FORCE_CPU
        self._prev = _FORCE_CPU
        _FORCE_CPU = True
        return self

    def __exit__(self, *exc):
        global _FORCE_CPU
        _FORCE_CPU = self._prev
        return False


def _cupy():
    """Return the cupy module iff a usable GPU is present, else None (probed once, cached)."""
    global _CUPY
    if _CUPY is None:
        try:
            import cupy as cp
            cp.cuda.runtime.getDevice()          # fail fast if there is no usable device / driver
            _CUPY = cp
        except Exception:
            _CUPY = False
    return _CUPY or None


def _have_pyfftw() -> bool:
    import importlib.util
    return importlib.util.find_spec("pyfftw") is not None


def select_backend(decisive: bool = False) -> str:
    """Resolve the backend name ('scipy' | 'pyfftw' | 'cupy') for one NCC call."""
    if _FORCE_CPU or decisive:
        return "scipy"
    pref = os.environ.get("PFLM_ACCEL", "auto").strip().lower()
    if pref in ("cpu", "scipy", ""):
        return "scipy"
    if pref == "pyfftw":
        return "pyfftw" if _have_pyfftw() else "scipy"
    if pref in ("cupy", "gpu"):
        if _cupy() is not None:
            return "cupy"
        return "pyfftw" if _have_pyfftw() else "scipy"
    # "auto" (and anything unrecognised): stay on the deterministic CPU path.  GPU/pyfftw are opt-in
    # until the confirm-on-CPU detection wiring lands, so the default never perturbs a measurement.
    return "scipy"


def _ncc_scipy(img, tpl):
    """BIT-IDENTICAL to the historical register._pattern_ncc (the reproducibility reference)."""
    from scipy.signal import fftconvolve
    t = tpl.astype(np.float64)
    t0 = t - t.mean()
    tnorm = float(np.sqrt((t0 * t0).sum()))
    ones = np.ones_like(t)
    if tnorm < 1e-9:
        Hi, Wi = img.shape; Ht, Wt = t.shape
        return np.zeros((Hi + Ht - 1, Wi + Wt - 1))
    num = fftconvolve(img, t0[::-1, ::-1], mode="full")            # sum (img * zero-mean tpl)
    s1 = fftconvolve(img, ones[::-1, ::-1], mode="full")           # local sum of img
    s2 = fftconvolve(img * img, ones[::-1, ::-1], mode="full")     # local sum of img^2
    n = float(t.size)
    denom = np.sqrt(np.maximum(s2 - s1 * s1 / n, 0.0)) * tnorm
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 1e-9, num / denom, 0.0)


def _ncc_pyfftw(img, tpl):
    """Same float64 maths as _ncc_scipy, but routed through the (multithreaded) FFTW backend."""
    import scipy.fft as _sfft
    import pyfftw
    import pyfftw.interfaces.scipy_fft as _pf
    pyfftw.interfaces.cache.enable()
    with _sfft.set_backend(_pf):
        return _ncc_scipy(img, tpl)


def _ncc_cupy(img, tpl):
    """GPU NCC in float32 for coarse search; returns a host float64 array.  Localization only."""
    cp = _cupy()
    from cupyx.scipy.signal import fftconvolve as _cfft
    t = cp.asarray(tpl, dtype=cp.float32)
    t0 = t - t.mean()
    tnorm = float(cp.sqrt((t0 * t0).sum()))
    if tnorm < 1e-9:
        Hi, Wi = img.shape; Ht, Wt = tpl.shape
        return np.zeros((Hi + Ht - 1, Wi + Wt - 1))
    im = cp.asarray(img, dtype=cp.float32)
    ones = cp.ones_like(t)
    num = _cfft(im, t0[::-1, ::-1], mode="full")
    s1 = _cfft(im, ones[::-1, ::-1], mode="full")
    s2 = _cfft(im * im, ones[::-1, ::-1], mode="full")
    n = float(t.size)
    denom = cp.sqrt(cp.maximum(s2 - s1 * s1 / n, 0.0)) * tnorm
    with np.errstate(divide="ignore", invalid="ignore"):
        out = cp.where(denom > 1e-9, num / denom, 0.0)
    return cp.asnumpy(out).astype(np.float64)


def pattern_ncc(img, tpl, *, decisive: bool = False):
    """Normalized cross-correlation (FULL mode) via the selected backend.

    ``decisive=True`` forces the deterministic CPU float64 path for calls whose peak value chooses a
    reflection/rotation.  Any GPU/pyfftw failure falls back to the CPU path so a run never dies for
    lack of a backend.
    """
    backend = select_backend(decisive)
    if backend == "cupy":
        try:
            return _ncc_cupy(img, tpl)
        except Exception:
            pass                                   # GPU hiccup -> deterministic CPU fallback
    if backend == "pyfftw":
        try:
            return _ncc_pyfftw(img, tpl)
        except Exception:
            pass
    return _ncc_scipy(img, tpl)
