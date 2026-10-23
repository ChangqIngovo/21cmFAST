"""Source-merged angular photon-counting transport research probe.

This is an explicit first-order discrete-ordinate finite-volume transport solver,
not a replacement ES criterion. It deliberately keeps the physics/cost tradeoff
visible: positive directional streaming, local capacity absorption, no source
count in the transport cost, but finite-angle and numerical-diffusion errors.
The source array is a photon *increment*, not a cumulative emission history.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

_C = r"""
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
int transport(int n, int nd, int steps, int emission_steps, int threads,
              int periodic, double dt, const double *directions,
              const double *weights, const double *source,
              const double *capacity, double *absorbed,
              double *radiation, double *by_angle, double *escaped) {
    size_t cells=(size_t)n*n*n, elements=cells*nd;
    double *next=calloc(elements,sizeof(double));
    if (!next) return 1;
    memset(absorbed,0,cells*sizeof(double));
    memset(radiation,0,elements*sizeof(double));
    memset(by_angle,0,nd*sizeof(double));
    memset(escaped,0,nd*sizeof(double));
    for (int step=0; step<=steps; ++step) {
        #pragma omp parallel for num_threads(threads) reduction(+:by_angle[:nd])
        for (size_t j=0;j<cells;++j) {
            double total=0.;
            for (int a=0;a<nd;++a) {
                if (step<emission_steps)
                    radiation[j*nd+a]+=source[j]*weights[a]/emission_steps;
                total+=radiation[j*nd+a];
            }
            double take=fmin(fmax(0.,capacity[j]-absorbed[j]),total);
            double fraction=total>0 ? take/total : 0.;
            absorbed[j]+=take;
            for (int a=0;a<nd;++a) {
                double delta=radiation[j*nd+a]*fraction;
                radiation[j*nd+a]-=delta;
                by_angle[a]+=delta;
            }
        }
        if(step==steps) break;
        #pragma omp parallel for num_threads(threads) reduction(+:escaped[:nd])
        for (size_t j=0;j<cells;++j) {
            int pos[3]={(int)(j/(n*n)),(int)((j/n)%n),(int)(j%n)};
            size_t stride[3]={(size_t)n*n,(size_t)n,1};
            for(int a=0;a<nd;++a) {
                double out=0., incoming=0.;
                for(int axis=0;axis<3;++axis) {
                    double velocity=directions[3*a+axis];
                    double coefficient=fabs(velocity)*dt;
                    out+=coefficient;
                    int shift=velocity>=0 ? -1 : 1;
                    int coord=pos[axis]+shift;
                    if(coord>=0 && coord<n) {
                        size_t k=(size_t)((long long)j+(long long)shift*stride[axis]);
                        incoming+=coefficient*radiation[k*nd+a];
                    } else if(periodic) {
                        int wrap=shift<0 ? n-1 : -(n-1);
                        size_t k=(size_t)((long long)j+(long long)wrap*stride[axis]);
                        incoming+=coefficient*radiation[k*nd+a];
                    }
                    if(!periodic && ((velocity>0 && pos[axis]==n-1) ||
                                      (velocity<0 && pos[axis]==0)))
                        escaped[a]+=coefficient*radiation[j*nd+a];
                }
                next[j*nd+a]=(1.-out)*radiation[j*nd+a]+incoming;
            }
        }
        memcpy(radiation,next,elements*sizeof(double));
    }
    free(next);
    return 0;
}
"""

_LIB = None


def _library():
    global _LIB
    if _LIB is None:
        token = hashlib.sha256(_C.encode()).hexdigest()[:16]
        directory = Path(tempfile.gettempdir()) / f"pc_sn_probe_{token}"
        directory.mkdir(exist_ok=True)
        source, binary = directory / "transport.c", directory / "transport.so"
        if not binary.exists():
            source.write_text(_C)
            subprocess.run(
                [os.environ.get("CC", "gcc"), "-O3", "-std=c99", "-fopenmp",
                 "-shared", "-fPIC", str(source), "-o", str(binary), "-lm"],
                check=True,
            )
        _LIB = ctypes.CDLL(str(binary))
        ptr = ctypes.POINTER(ctypes.c_double)
        _LIB.transport.argtypes = [ctypes.c_int]*6 + [ctypes.c_double] + [ptr]*8
        _LIB.transport.restype = ctypes.c_int
    return _LIB


def angular_directions(count):
    """Return antipodal, equal-weight Fibonacci quadrature directions."""
    if count < 2 or count % 2:
        raise ValueError("An even direction count >=2 is required")
    index = np.arange(count//2, dtype=float)
    z = (index+.5)/(count//2)
    angle = np.pi*(3-np.sqrt(5))*index
    r = np.sqrt(1-z*z)
    upper = np.column_stack((r*np.cos(angle), r*np.sin(angle), z))
    directions = np.concatenate((upper, -upper))
    return directions, np.full(count, 1/count)


def solve(source, capacity, *, steps=64, directions=24, weights=None,
          emission_steps=1, cfl=.85, threads=4, periodic=True, _backend=None):
    """Transport an injected photon increment with explicit remaining radiation.

    Cell size and light speed are one; elapsed time is `steps*dt`. Photons
    injected during the first `emission_steps` are absorbed locally up to the
    remaining cell capacity, with incoming channels sharing each absorption in
    proportion to their local photon populations. Neutral cells are infinitely
    opaque until their finite ionization capacity has been exhausted. Already
    ionized cells have zero remaining capacity. No recombinations are modeled.
    """
    source = np.ascontiguousarray(source, dtype=np.float64)
    capacity = np.ascontiguousarray(capacity, dtype=np.float64)
    if source.ndim != 3 or len(set(source.shape)) != 1 or capacity.shape != source.shape or source.shape[0] < 2:
        raise ValueError("Matched cubic grids are required")
    if not (np.isfinite(source).all() and np.isfinite(capacity).all()):
        raise ValueError("Finite input counts required")
    if np.any(source < 0) or np.any(capacity < 0):
        raise ValueError("Nonnegative counts required")
    if not 1 <= emission_steps <= steps or threads < 1 or not 0 < cfl <= 1:
        raise ValueError("Invalid stepping options")
    if np.isscalar(directions):
        directions, default_weights = angular_directions(int(directions))
    else:
        directions = np.asarray(directions, dtype=float)
        default_weights = np.full(len(directions), 1/len(directions))
    directions = np.ascontiguousarray(directions, dtype=np.float64)
    weights = np.ascontiguousarray(default_weights if weights is None else weights,
                                 dtype=np.float64)
    if directions.shape != (len(weights), 3) or not np.isfinite(directions).all():
        raise ValueError("Direction array does not match weights")
    if not np.allclose(np.linalg.norm(directions, axis=1), 1, rtol=1e-12, atol=1e-12):
        raise ValueError("Directions must be unit vectors")
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1, rtol=0, atol=2e-14):
        raise ValueError("Weights must form a nonnegative partition of one")
    dt = cfl/np.max(np.abs(directions).sum(axis=1))
    absorbed = np.zeros_like(source)
    radiation = np.zeros((*source.shape, len(weights)))
    by_angle, escaped = np.zeros(len(weights)), np.zeros(len(weights))
    def ptr(array):
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    lib = _library() if _backend is None else _backend
    started = time.perf_counter()
    status = lib.transport(source.shape[0], len(weights), steps, emission_steps,
                           threads, int(periodic), dt, ptr(directions), ptr(weights),
                           ptr(source), ptr(capacity), ptr(absorbed), ptr(radiation),
                           ptr(by_angle), ptr(escaped))
    elapsed = time.perf_counter()-started
    if status:
        raise MemoryError("Transport temporary allocation failed")
    emitted = float(source.sum(dtype=np.longdouble))
    pending = radiation.sum(axis=(0, 1, 2), dtype=np.longdouble)
    residual = emitted-np.sum(absorbed, dtype=np.longdouble)-pending.sum()-escaped.sum()
    return {
        "absorbed": absorbed, "radiation": radiation, "directions": directions,
        "weights": weights, "absorbed_by_angle": by_angle, "escaped_by_angle": escaped,
        "diagnostics": {
            "emitted": emitted, "absorbed": float(np.sum(absorbed, dtype=np.longdouble)),
            "pending": float(pending.sum()), "escaped": float(escaped.sum()),
            "ledger_residual": float(residual),
            "maximum_angle_residual": float(np.max(np.abs(emitted*weights-by_angle-pending-escaped))),
            "seconds": elapsed, "steps": steps, "dt": float(dt), "elapsed_time": float(steps*dt),
            "angles": len(weights), "threads": threads,
            "direction_cell_updates": int(steps*source.size*len(weights)),
            "working_array_bytes": int(2*radiation.nbytes+2*source.nbytes),
        },
    }


def overlap_quadrature(counts=(12, 24, 48, 96, 192, 384), orientations=1000):
    """Measure the one-eighth overlap cone, not the whole apparent second sphere."""
    rng = np.random.default_rng(417)
    axes = rng.normal(size=(orientations, 3))
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    result = []
    for count in counts:
        dirs, weights = angular_directions(count)
        fractions = ((axes@dirs.T) > .75)@weights
        result.append({"directions": count, "exact": .125,
                       "mean": float(fractions.mean()),
                       "max_absolute_error": float(np.max(abs(fractions-.125))),
                       "rms_absolute_error": float(np.sqrt(np.mean((fractions-.125)**2)))})
    return result
