# Research figures

`plot_geometry_summary.py` reads saved JSON/NPZ results only and regenerates
the PNG and PDF figures. It does not execute a transport solver. Each figure
has a `*_sources.json` file recording source hashes and scope.

## `overlap_photon_deposition`

Two initially transparent bubbles of radius 12 cells are separated by 18 cells.
An isotropic 100-photon increment originates at the left centre. The two panels
show the absorbed increment projected along z, using the same logarithmic colour
scale and the same original voxel mask. The dashed line is the initial midplane
boundary, not the boundary of the projected absorption. Consequently absorption
from the front/back surfaces appears inside the dashed contour in projection.

The left panel is the research SCRIPT-like radial allocator, not a claim about
all SCRIPT implementations. It distributes the increment through radial shells
using remaining capacities. The right panel follows the first opaque voxel on
each fixed direction. At K=12,288 its event solution agrees with an independent
ordinary DDA histogram to normalized L1 difference 6.5e-16. The largest voxel
receives only 0.114 photons, below its unit capacity, so this increment does not
advance any full voxel front: fixed first-absorption geometry is sufficient.

Both allocations conserve the total photons, but their absorption centroids
shift in opposite directions: about -1.31 versus +1.23 cells from the source.
The continuum infinitesimal-increment value is +1.309 cells; the remaining
event difference reflects the voxel boundary. Exactly 12.5% of the emitted
directions leave the first sphere through the overlap cone. This fraction must
not be inferred by counting absorption voxel centres in the cone; discretized
first-hit surfaces and projection alter that classification.

## `geometry_checks`

Panel a compares normalized L1 error against the continuous sphere integrated
over voxels. Event/DDA uses K=ceil(16 pi R²); MUSCL uses K=96, with an additional
K=384 point at R=12. Angular resolutions and emission-time approximations differ.
The plot demonstrates geometric convergence and residual error; it is not an
equal-cost or fully matched-operator comparison. Event references use 48-by-48
midpoint x/y quadrature with exact z integration; MUSCL references use 24-by-24.
The measured 24/48 reference discrepancy is at most 1.02e-4 of Q, much smaller
than the displayed errors.

Panel b measures error when finite antipodal Fibonacci directions sample the
analytic 1/8 overlap cone. RMS and largest sampled errors use 1000 orientations.
The largest sampled errors are not confidence intervals or proven all-orientation
bounds. No finite direction count is presented as an exact continuum transport
solution.
