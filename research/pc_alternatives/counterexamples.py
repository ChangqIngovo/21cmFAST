"""Finite witnesses separating photon accounting, allocation, and transport.

These are verification problems, not a proposed reionization solver. All numbers
represent photons/neutral atoms in a static, monochromatic, no-recombination
sharp-front model; explicit beam examples exclude scattering and diffuse light.
Run with Python + NumPy + SciPy. A linear program verifies allocation feasibility
only: its optimizer is not claimed to solve radiative transfer.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


def independent_barriers() -> dict:
    """Verify independent passing spheres can jointly overspend one donor."""
    # Embed this line in 3D; empty cells have zero source and capacity. Each
    # target's radius-one ball contains only its own target capacity.
    source_x = np.array([0.0, 10.0])
    source = np.array([1.25, 1.75])
    target_x = np.array([-1.0, 1.0, 10.0])
    capacity = np.ones(3)
    reachable = np.abs(source_x[:, None] - target_x[None, :]) <= 1.0
    barrier_supply = source @ reachable
    assert np.all(barrier_supply > capacity)
    assert source.sum() == capacity.sum()

    # Maximize total nonnegative absorption under source and target capacities.
    constraints = np.zeros((source.size + capacity.size, reachable.size))
    for s, j in itertools.product(range(source.size), range(capacity.size)):
        constraints[s, s * capacity.size + j] = 1
        constraints[source.size + j, s * capacity.size + j] = 1
    result = linprog(
        -np.ones(reachable.size),
        A_ub=constraints,
        b_ub=np.r_[source, capacity],
        bounds=[(0, None if edge else 0) for edge in reachable.flat],
        method="highs",
    )
    assert result.success
    allocation = result.x.reshape(reachable.shape)
    assert np.isclose(allocation.sum(), 2.25)

    deficiencies = []
    for bits in itertools.product((False, True), repeat=capacity.size):
        chosen = np.array(bits)
        if not np.any(chosen):
            continue
        neighbors = np.any(reachable[:, chosen], axis=1)
        deficiency = capacity[chosen].sum() - source[neighbors].sum()
        if deficiency > 0:
            deficiencies.append(
                {
                    "targets": np.flatnonzero(chosen).tolist(),
                    "demand_minus_accessible_budget": float(deficiency),
                }
            )
    assert deficiencies == [{"targets": [0, 1], "demand_minus_accessible_budget": 0.75}]
    return {
        "source_x": source_x.tolist(),
        "source_budget": source.tolist(),
        "target_x": target_x.tolist(),
        "target_capacity": capacity.tolist(),
        "radius": 1.0,
        "all_independent_barriers_strictly_pass": True,
        "global_budget_equals_capacity": True,
        "reachable": reachable.tolist(),
        "barrier_supply": barrier_supply.tolist(),
        "max_allocatable": float(allocation.sum()),
        "unavoidable_unfilled_capacity": float(capacity.sum() - allocation.sum()),
        "unavoidable_unused_budget": float(source.sum() - allocation.sum()),
        "violated_subset_conditions": deficiencies,
    }


def beam_precedence() -> dict:
    """Exhibit allocation feasibility without physical absorption ordering."""
    budget = 1.0
    capacity = np.ones(2)
    simultaneous = budget * capacity / capacity.sum()
    physical = np.array([1.0, 0.0])
    assert np.all(simultaneous <= capacity)
    assert simultaneous.sum() == budget
    assert simultaneous[1] > 0
    assert simultaneous[0] < capacity[0]
    return {
        "model": "one collimated beam; opaque sharp fronts; foreground then background",
        "capacity": capacity.tolist(),
        "budget": budget,
        "capacity_proportional_allocation": simultaneous.tolist(),
        "physical_allocation": physical.tolist(),
        "source_and_target_constraints_pass": True,
        "background_absorption_before_foreground_exhaustion": float(simultaneous[1]),
    }


def angular_budget() -> dict:
    """Exhibit isotropic sector overspending even at a single radius."""
    # First intercepting absorbers at the same radius in two equal-solid-angle
    # cones. Q/2 is emitted into each cone. Thin radial shells do not fix this.
    budget = 2.0
    capacity = np.array([1.0, 100.0])
    simultaneous = budget * capacity / capacity.sum()
    angular_quota = np.ones(2)
    physical = np.minimum(capacity, angular_quota)
    assert np.all(simultaneous <= capacity)
    assert simultaneous.sum() == budget
    assert simultaneous[1] > angular_quota[1]
    return {
        "model": "isotropic source; two equal-angle cones; first absorber at equal radius",
        "capacity": capacity.tolist(),
        "budget": budget,
        "per_cone_emission": angular_quota.tolist(),
        "capacity_proportional_allocation": simultaneous.tolist(),
        "physical_allocation": physical.tolist(),
        "source_and_target_constraints_pass": True,
        "largest_cone_quota_excess": float(np.max(simultaneous - angular_quota)),
    }


def angular_moment_loss() -> dict:
    """Compare equal low moments with distinct angular boundary responses."""
    directions_a = np.array([[1, 0, 0], [-1, 0, 0]], dtype=float)
    directions_b = np.array([[0, 1, 0], [0, -1, 0]], dtype=float)
    moments = [
        {
            "number_density_E": float(len(directions)),
            "number_flux_F_over_c": directions.sum(axis=0).tolist(),
            "plus_x_face_incident_flux_over_c": float(
                np.maximum(directions[:, 0], 0).sum()
            ),
            "second_moment_P": np.einsum("ni,nj->ij", directions, directions).tolist(),
        }
        for directions in (directions_a, directions_b)
    ]
    assert moments[0]["number_density_E"] == moments[1]["number_density_E"]
    assert moments[0]["number_flux_F_over_c"] == moments[1]["number_flux_F_over_c"]
    assert moments[0]["plus_x_face_incident_flux_over_c"] == 1
    assert moments[1]["plus_x_face_incident_flux_over_c"] == 0
    return {
        "opposed_x_beams": moments[0],
        "opposed_y_beams": moments[1],
        "consequence": "E,F alone cannot distinguish these angular boundary responses",
    }


def radial_statistics_loss() -> dict:
    """Rotate an opaque cloud while retaining all one-center radial counts."""
    # These are two distinct spatial fields; their full FFTs are distinguishable.
    shape = (9, 9, 9)
    center = np.array([4, 4, 4])
    cloud_positions = (np.array([5, 4, 4]), np.array([4, 5, 4]))
    target = np.array([6, 4, 4])
    distance = np.sqrt(
        sum((x - c) ** 2 for x, c in zip(np.indices(shape), center, strict=True))
    )
    profiles = []
    for cloud in cloud_positions:
        density = np.zeros(shape)
        density[tuple(cloud)] = 100
        density[tuple(target)] = 1
        profiles.append([float(density[distance <= r].sum()) for r in range(9)])
    assert profiles[0] == profiles[1]
    return {
        "source": center.tolist(),
        "target": target.tolist(),
        "opaque_cloud_positions": [p.tolist() for p in cloud_positions],
        "source_centered_radial_cumulative_capacity": profiles,
        "same_radial_statistics": True,
        "target_central_ray_blocked": [True, False],
        "scope": "one-center radial aggregates; not full all-position FFT information",
    }


def nonpositive_sharp_k() -> dict:
    """Show why a sharp-k filter is not a probability transition kernel."""
    n = 32
    wave = np.fft.fftfreq(n) * n
    k2 = sum(component**2 for component in np.meshgrid(wave, wave, wave, indexing="ij"))
    multiplier = (k2 <= 4**2).astype(float)
    kernel = np.fft.ifftn(multiplier).real
    assert np.isclose(kernel.sum(), 1.0)
    assert kernel.min() < -1e-5
    return {
        "grid": n,
        "cutoff_in_fundamental_modes": 4,
        "kernel_sum": float(kernel.sum()),
        "kernel_min": float(kernel.min()),
        "kernel_negative_mass": float(-kernel[kernel < 0].sum()),
        "negative_voxels": int(np.count_nonzero(kernel < 0)),
        "consequence": "a sharp-k density filter is not a nonnegative photon transfer kernel",
    }


def budget_does_not_fix_power() -> dict:
    """Produce different powers from two equal-budget ionization maps."""
    adjacent = np.array([1, 1, 0, 0], dtype=float)
    alternating = np.array([1, 0, 1, 0], dtype=float)
    p_adjacent = np.abs(np.fft.fft(adjacent)) ** 2
    p_alternating = np.abs(np.fft.fft(alternating)) ** 2
    assert adjacent.sum() == alternating.sum() == 2
    assert np.isclose(p_adjacent[1], 2)
    assert p_alternating[1] == 0
    return {
        "absorbed_photons_each": 2.0,
        "capacity_per_cell": 1.0,
        "adjacent_ionization": adjacent.tolist(),
        "alternating_ionization": alternating.tolist(),
        "unnormalized_power_k1": [float(p_adjacent[1]), float(p_alternating[1])],
        "consequence": "global photon closure alone predicts neither sign nor scale of power changes",
    }


def two_spheres() -> dict:
    """Disentangle the overlap cap, apparent solid angle, and overlap volume."""
    radius, separation = 1.0, 1.5
    surface_mu = separation / (2 * radius)
    tangent_mu = np.sqrt(1 - (radius / separation) ** 2)
    surface_fraction = (1 - surface_mu) / 2
    apparent_fraction = (1 - tangent_mu) / 2
    samples = 1_000_000
    mu = -1 + 2 * (np.arange(samples) + 0.5) / samples
    # A surface point R*n lies within B iff mu >= d/(2R).
    measured_surface = float(np.mean(mu >= surface_mu))
    # A positive ray intersects B iff mu >= sqrt(1-(R/d)^2).
    measured_apparent = float(np.mean(mu >= tangent_mu))
    overlap_over_single_volume = (
        (4 * radius + separation) * (2 * radius - separation) ** 2 / (16 * radius**3)
    )
    assert abs(measured_surface - surface_fraction) <= 1 / samples
    assert abs(measured_apparent - apparent_fraction) <= 1 / samples
    assert surface_fraction == 1 / 8
    assert overlap_over_single_volume == 11 / 128
    return {
        "R": radius,
        "d": separation,
        "fraction_A_surface_inside_B": float(surface_fraction),
        "fraction_all_rays_intersecting_B": float(apparent_fraction),
        "equal_area_direction_samples": samples,
        "sampled_surface_fraction": measured_surface,
        "sampled_apparent_fraction": measured_apparent,
        "overlap_volume_over_one_sphere": float(overlap_over_single_volume),
        "overlap_volume_over_sum_of_two_spheres": float(overlap_over_single_volume / 2),
        "scope": "geometry of fixed spheres; not automatically a dynamical photon allocation",
    }


def certified_nearly_spherical_front() -> dict:
    """Verify the atom-error certificate against an exact anisotropic front."""
    # Four equal-solid-angle sectors with radially constant density. This is
    # an exact solvable anisotropic front, not an angular transport benchmark.
    density = np.array([0.9, 1.0, 1.1, 1.0])
    solid_angle = np.full(4, np.pi)
    trial_radius = 1.0
    atoms_per_steradian = density * trial_radius**3 / 3
    total_photons = float(solid_angle @ atoms_per_steradian)
    photon_quota_per_steradian = total_photons / (4 * np.pi)
    exact_radius = (3 * photon_quota_per_steradian / density) ** (1 / 3)
    exact_atoms_per_steradian = density * exact_radius**3 / 3
    epsilon = float(
        np.max(np.abs(atoms_per_steradian / photon_quota_per_steradian - 1))
    )
    morphology_l1 = float(
        solid_angle @ np.abs(atoms_per_steradian - exact_atoms_per_steradian)
    )
    assert np.allclose(exact_atoms_per_steradian, photon_quota_per_steradian)
    assert morphology_l1 <= epsilon * total_photons + 1e-14
    assert np.isclose(morphology_l1 / total_photons, 0.05)
    return {
        "density_in_four_equal_angle_sectors": density.tolist(),
        "trial_spherical_radius": trial_radius,
        "exact_directional_radii": exact_radius.tolist(),
        "photons": total_photons,
        "uniform_relative_directional_demand_bound": epsilon,
        "actual_relative_atom_morphology_l1": morphology_l1 / total_photons,
        "proved_relative_atom_morphology_l1_upper_bound": epsilon,
        "scope": "single isolated source; known per-direction demand; no recombination",
    }


def main() -> None:
    """Run every finite witness and write machine-readable results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("counterexamples_results.json"),
    )
    args = parser.parse_args()
    tests = [
        independent_barriers,
        beam_precedence,
        angular_budget,
        angular_moment_loss,
        radial_statistics_loss,
        nonpositive_sharp_k,
        budget_does_not_fix_power,
        two_spheres,
        certified_nearly_spherical_front,
    ]
    results = {test.__name__: test() for test in tests}
    results["scope"] = "Finite mathematical witnesses, not simulations of Qin20"
    results["all_assertions_passed"] = True
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"{len(tests)} counterexample/geometry probes passed; wrote {args.output}")  # noqa: T201


if __name__ == "__main__":
    main()
