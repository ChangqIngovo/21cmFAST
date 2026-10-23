"""Small independent scientific checks for the geometry-validation harness."""
import numpy as np

from validation_geometry import continuous_union_exit,sphere_measure,tagged_bubble_measure


def test_continuous_two_ball_first_exit_detects_neutral_gap():
    mu=np.array([1.,-1.,.74,.76])
    dirs=np.column_stack((mu,np.sqrt(1-mu*mu),np.zeros_like(mu)))
    length,cone=continuous_union_exit(12,dirs)
    assert np.allclose(length[:2],[30,12])
    # The ray can intersect B later yet first encounter neutral gas at A's
    # surface. It must not jump across that neutral gap.
    assert length[2]==12 and length[3]>12
    assert np.array_equal(cone,[True,False,False,True])


def test_fixed_grid_angular_refinement_reduces_continuous_sphere_error():
    low=sphere_measure(8,48);high=sphere_measure(8,3072)
    assert high['voxel_sphere_l1_per_photon']<low['voxel_sphere_l1_per_photon']
    assert high['escaped_fraction']==0
    assert high['reference']['quadrature24_48_l1_per_photon']<2e-4


def test_fixed_high_angle_spatial_refinement_reduces_voxel_front_error():
    coarse=sphere_measure(4,12288);fine=sphere_measure(12,12288)
    assert fine['voxel_sphere_l1_per_photon']<coarse['voxel_sphere_l1_per_photon']


def test_tagged_overlap_spatial_refinement_and_event_histogram():
    coarse=tagged_bubble_measure(4,12288)
    fine=tagged_bubble_measure(24,12288,verify_event=True)
    assert fine['mean_absolute_front_distance_error_over_R']<coarse['mean_absolute_front_distance_error_over_R']
    exact=fine['exact_continuum_centroid_x_over_R']
    assert abs(fine['first_front_centroid_over_R'][0]-exact)<abs(coarse['first_front_centroid_over_R'][0]-exact)
    assert fine['event_vs_frozen_histogram_linf']<1e-12
    assert fine['B_exit_fraction_lower_bound']<=fine['B_exit_fraction_upper_bound']
