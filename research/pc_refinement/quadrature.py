"""Cost-neutral angular rules for the existing shared photon event core.

All rules retain equal positive photon weights and exactly the same angular
counts and transport equations. A private copy of the Python interface owns
the direction factory; neither the production/research baseline module nor its
C++ core is modified. The lock prevents simultaneous factory changes.

``moment-design`` uses an equal-weight 24-direction spherical 5-design and a
48-direction spherical 7-design. Higher counts combine deterministic rotated
copies of the 48-point rule. Exact smooth angular moments do not certify
accuracy for discontinuous first-absorber maps.
"""
from __future__ import annotations

from functools import lru_cache
import importlib.util
from itertools import permutations, product
from pathlib import Path
from threading import RLock

import numpy as np
from scipy.spatial.transform import Rotation


BASELINE = Path(__file__).resolve().parents[1] / "pc_reconsidered" / "hybrid.py"
_spec = importlib.util.spec_from_file_location("_pc_quadrature_backend", BASELINE)
_backend = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_backend)
_original_directions = _backend.directions
_lock = RLock()
METHODS = ("fibonacci", "antipodal", "moment-design", "rotated-fibonacci",
           "low-order-design", "cubic-equal-area", "voronoi-weights", "moment-weights")


def _orbit(squared):
    """Full signed permutation orbit, with duplicate coordinates removed."""
    directions = []
    for p in sorted(set(permutations(np.sqrt(squared)))):
        for signs in product((-1., 1.), repeat=3):
            directions.append(np.asarray(p) * signs)
    return np.asarray(directions)


def _rotation(index):
    """Deterministic Haar-coordinate Kronecker samples; no RNG dependency."""
    if index == 0:
        return np.eye(3)
    u, v, w = np.mod(index * np.sqrt([2., 3., 5.]), 1.)
    quaternion = [np.sqrt(1-u)*np.sin(2*np.pi*v),
                  np.sqrt(1-u)*np.cos(2*np.pi*v),
                  np.sqrt(u)*np.sin(2*np.pi*w),
                  np.sqrt(u)*np.cos(2*np.pi*w)]
    return Rotation.from_quat(quaternion).as_matrix()


def _fundamental_triangle(count):
    """Equal-area Hammersley nodes in x>=y>=z>=0, area pi/12.

    With n=(cos(theta),sin(theta)cos(phi),sin(theta)sin(phi)),
    phi in[0,pi/4] and theta_max=atan(1/cos(phi)). The azimuth
    CDF is phi-asin(sin(phi)/sqrt(2)). Conditional cos(theta)
    is uniform. Every node represents one equal angular weight.
    """
    u = (np.arange(count) + .5) / count
    bits = max(1, int(np.ceil(np.log2(count))))
    if count & (count-1):
        v = np.mod(np.arange(count)*(np.sqrt(5.)-1)/2+.5, 1.)
    else:
        reverse = np.zeros(count, dtype=int)
        values = np.arange(count)
        for _ in range(bits):
            reverse = 2*reverse + values % 2
            values //= 2
        v = 1-(reverse+.5)/count
    phi = u*np.pi/4
    target = u*np.pi/12
    for _ in range(8):
        residual = phi-np.arcsin(np.sin(phi)/np.sqrt(2))-target
        derivative = 1-np.cos(phi)/np.sqrt(1+np.cos(phi)**2)
        phi -= residual/derivative
    cos_max = np.cos(phi)/np.sqrt(1+np.cos(phi)**2)
    mu = 1-v*(1-cos_max)
    return np.column_stack((mu, np.sqrt(1-mu**2)*np.cos(phi),
                            np.sqrt(1-mu**2)*np.sin(phi)))


@lru_cache(maxsize=100)
def directions(count, method="fibonacci"):
    """Return deterministic, equal-weight unit directions."""
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("count must be a positive integer")
    if method not in METHODS:
        raise ValueError(f"Unknown angular rule {method!r}")
    if method.endswith("-weights"):
        return directions(count, "fibonacci")
    if method == "low-order-design":
        return directions(count, "moment-design" if count <= 48 else "fibonacci")
    if method == "cubic-equal-area":
        if count == 24:
            return directions(count, "moment-design")
        if count % 48:
            raise ValueError("Cubic equal-area rule requires 24 or a multiple of 48 angles")
        values = np.concatenate([_orbit(point**2) for point in _fundamental_triangle(count//48)])
    elif method == "fibonacci":
        values = _original_directions(count)
    elif method == "rotated-fibonacci":
        values = _original_directions(count) @ _rotation(1).T
    elif method == "antipodal":
        if count % 2:
            raise ValueError("Antipodal quadrature requires an even count")
        j = np.arange(count // 2)
        mu = (j + .5) / (count // 2)
        phi = j * np.pi * (3 - np.sqrt(5.))
        half = np.column_stack((mu, np.sqrt(1-mu**2)*np.cos(phi),
                                np.sqrt(1-mu**2)*np.sin(phi)))
        values = np.concatenate((half, -half))
    else:
        if count == 24:
            # p+2q=1 and (p^2+2q^2)/3=1/5 give exact degree-4 moments.
            p = (1 + np.sqrt(1.6)) / 3
            values = _orbit((p, (1-p)/2, (1-p)/2))
        elif count % 48 == 0:
            # For squared coordinates a,b,c, sum=1, sum pairwise=1/5,
            # product=1/105 imply the isotropic moments through degree six.
            roots = np.sort(np.roots([1., -1., .2, -1/105.]).real)
            base = _orbit(roots)
            values = np.concatenate([base @ _rotation(i).T for i in range(count // 48)])
        else:
            raise ValueError("Moment design requires 24 or a multiple of 48 angles")
    values = np.ascontiguousarray(values, dtype=np.float64)
    if values.shape != (count, 3):
        raise RuntimeError("Quadrature constructed the wrong number of directions")
    if not np.allclose(np.linalg.norm(values, axis=1), 1., atol=2e-15, rtol=0):
        raise RuntimeError("Quadrature directions are not normalized")
    values.flags.writeable = False
    return values


def solve(source, capacity, initial_ionized=None, *, method="fibonacci", **options):
    """Apply a new angular rule with the unchanged source/atom event solver."""
    if method not in METHODS:
        raise ValueError(f"Unknown angular rule {method!r}")
    if method.endswith("-weights"):
        from quadrature_weights import solve as solve_weighted
        return solve_weighted(source, capacity, initial_ionized, method=method, **options)
    with _lock:
        _backend.directions = lambda count: directions(count, method)
        try:
            result = _backend.solve(source, capacity, initial_ionized, **options)
        finally:
            _backend.directions = _original_directions
    result["audit"]["quadrature"] = method
    return result


def warmup():
    """Compile once, outside all recorded measurements."""
    _backend.library()
    from quadrature_weights import library
    library()
