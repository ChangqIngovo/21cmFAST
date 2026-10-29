"""Positive species advection following the native Lagrangian mass packets.

The particle trajectories are prescribed by native 21cmFAST LPT. Their masses
are uniform cubes one native cell wide. Their volume overlap with native cells
is exactly its CIC convention. Between positions, overlap of each old cell and
each new cell pulled back along the packet translation determines the positive
transfer matrix. Its column/row sums are the old/new capacities. Applying the
same matrix to the old ion fraction conserves ions and preserves bounds without
resetting, clipping, or rescaling their counts. Interior material moves even
where the cloud density does not change.

Species fractions are piecewise constant within each old cell. This is a
first-order finite-volume species-advection discretization. Its time/grid
convergence must be measured. With no displacement the map is exactly the
identity, so merely splitting a stationary interval causes no mixing.

Positions are in *requested-grid* cell coordinates, and packet masses in mean
atom-count units of a requested-grid cell. Refinement by f requires positions
``p*f+(f-1)/2``, ``mass*f**3`` and ``cloud_width=f``. This integrates the same
physical cubes into smaller finite-volume cells. Block restriction reproduces
native CIC density to roundoff; refinement does not shrink the mass clouds or
introduce a new particle-discreteness scale.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
import hashlib
from pathlib import Path
import subprocess
import tempfile

import numpy as np

TOLERANCE = 2e-11


@lru_cache(maxsize=1)
def _library():
    source = Path(__file__).with_name("gas_kernel.cpp")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    directory = Path(tempfile.mkdtemp(prefix=f"causal_gas_{digest}_"))
    binary = directory / "gas.so"
    subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC",
                    str(source), "-o", str(binary)], check=True, capture_output=True)
    lib = ctypes.CDLL(str(binary))
    lib.gas_deposit.argtypes = [ctypes.c_int, ctypes.c_int64, ctypes.c_double] + [ctypes.c_void_p]*3
    lib.gas_deposit.restype = ctypes.c_int
    lib.gas_remap.argtypes = [ctypes.c_int, ctypes.c_int64, ctypes.c_double] + [ctypes.c_void_p]*5
    lib.gas_remap.restype = ctypes.c_int
    return lib


def _packets(mass, positions, dimension, cloud_width):
    if type(dimension) is not int or dimension < 2:
        raise ValueError("dimension must be an integer >= 2")
    if (isinstance(cloud_width, bool) or not np.isfinite(cloud_width)
            or not 1 <= cloud_width <= dimension):
        raise ValueError("cloud_width must lie between one requested cell and the box width")
    if np.iscomplexobj(mass) or np.iscomplexobj(positions):
        raise ValueError("Packet masses and coordinates must be real")
    mass = np.ascontiguousarray(mass, dtype=np.float64)
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    if (mass.ndim != 1 or positions.shape != (len(mass), 3)
            or not np.isfinite(mass).all() or np.any(mass < 0)
            or not np.isfinite(positions).all()):
        raise ValueError("Finite nonnegative packet masses and matching [particles,3] positions required")
    if mass.sum(dtype=np.longdouble) > np.finfo(float).max:
        raise ValueError("Packet mass exceeds the supported range")
    return mass, np.ascontiguousarray(np.remainder(positions, dimension))


def _relative(a, b, scale):
    return float(abs(np.longdouble(a) - np.longdouble(b))
                 / max(np.longdouble(scale), np.longdouble(np.finfo(float).tiny)))


def deposit(mass, positions, dimension, *, cloud_width=1.):
    """Integrate uniform mass clouds into periodic cells; width=1 is native CIC."""
    mass, positions = _packets(mass, positions, dimension, cloud_width)
    result = np.zeros((dimension,)*3, dtype=np.float64)
    if _library().gas_deposit(dimension, len(mass), cloud_width, mass.ctypes.data,
                              positions.ctypes.data, result.ctypes.data):
        raise RuntimeError("Native-packet CIC deposition failed")
    total = mass.sum(dtype=np.longdouble)
    residual = _relative(result.sum(dtype=np.longdouble), total, total)
    if not np.isfinite(result).all() or (result < 0).any() or residual > TOLERANCE:
        raise RuntimeError("CIC deposition failed its independent mass ledger")
    return result


def remap_ions(ionized, mass, old_positions, new_positions, dimension, *, cloud_width=1.):
    """Carry ions with a positive native-packet mass transfer, retaining all counts.

    Returns the new ``ionized`` and ``capacity`` grids and an independent audit.
    Initial ions must fit the capacity obtained from the supplied old packets.
    Mass and ion ledger errors are reported relative to total atoms and total
    initial ions respectively; empty ion inventories remain exactly zero.
    """
    mass, old = _packets(mass, old_positions, dimension, cloud_width)
    _, new = _packets(mass, new_positions, dimension, cloud_width)
    if np.iscomplexobj(ionized):
        raise ValueError("Ion counts must be real")
    ions = np.ascontiguousarray(ionized, dtype=np.float64)
    old_capacity = deposit(mass, old, dimension, cloud_width=cloud_width)
    new_capacity = deposit(mass, new, dimension, cloud_width=cloud_width)
    if (ions.shape != old_capacity.shape or not np.isfinite(ions).all()
            or np.any(ions < 0)
            or np.any(ions - old_capacity > TOLERANCE * old_capacity)):
        raise ValueError("Ionized counts must be finite, nonnegative, and within old packet capacity")
    fraction = np.divide(ions, old_capacity, out=np.zeros_like(ions), where=old_capacity > 0)
    unchanged = np.array_equal(old, new)
    updated = ions.copy() if unchanged else np.zeros_like(ions)
    if not unchanged and _library().gas_remap(
            dimension, len(mass), cloud_width, mass.ctypes.data, old.ctypes.data, new.ctypes.data,
            fraction.ctypes.data, updated.ctypes.data):
        raise RuntimeError("Conservative native-packet ion remapping failed")
    total_atoms = mass.sum(dtype=np.longdouble)
    initial_ions, final_ions = (a.sum(dtype=np.longdouble) for a in (ions, updated))
    ion_residual = _relative(initial_ions, final_ions, initial_ions)
    mass_residual = _relative(old_capacity.sum(dtype=np.longdouble),
                              new_capacity.sum(dtype=np.longdouble), total_atoms)
    bounded = (np.isfinite(updated).all() and not np.any(updated < 0)
               and not np.any(updated - new_capacity > TOLERANCE * new_capacity))
    if not bounded or max(ion_residual, mass_residual) > TOLERANCE:
        raise RuntimeError("Gas advection failed independent species/mass conservation or bounds")
    return {
        "ionized": updated, "capacity": new_capacity,
        "audit": {
            "passed": True, "mass_relative_residual": mass_residual,
            "ion_relative_residual": ion_residual, "initial_ionized": float(initial_ions),
            "final_ionized": float(final_ions), "total_atoms": float(total_atoms),
            "identity_for_zero_displacement": unchanged,
            "cloud_width_cells": float(cloud_width),
            "method": "translated mass-cloud cell overlaps; first-order species reconstruction",
        },
    }
