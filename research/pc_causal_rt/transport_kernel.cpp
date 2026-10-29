// Positive, conservative MUSCL/SSPRK2 discrete-ordinate photon transport.
// The carried ion and radiation states are never reset or globally rescaled.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <omp.h>
#include <vector>

namespace {
constexpr double light_speed = 2.99792458e10;
constexpr double ion_bound_tolerance = 2e-11;

double limited_slope(double left, double centre, double right) {
    const double a = centre - left;
    const double b = right - centre;
    if ((a <= 0 && b >= 0) || (a >= 0 && b <= 0)) return 0;
    const double magnitude = std::min({2 * std::abs(a), 2 * std::abs(b),
                                       0.5 * std::abs(a + b)});
    return std::copysign(magnitude, a);
}

std::int64_t neighbor(std::int64_t cell, int axis, int offset, int n) {
    const std::int64_t stride = axis == 0 ? static_cast<std::int64_t>(n) * n
                                         : (axis == 1 ? n : 1);
    const int coordinate = static_cast<int>((cell / stride) % n);
    const int next = (coordinate + offset + n) % n;
    return cell + static_cast<std::int64_t>(next - coordinate) * stride;
}

// Under sum(lambda)<=1/2, MC reconstruction and this unsplit Euler update
// preserve nonnegativity. Each upwind face has exactly the same value in its
// receiving and donating cell; the periodic sum is therefore conservative.
bool forward_euler(const double* source, double* destination, int n,
                   const double* direction, double factor) {
    const std::int64_t cells = static_cast<std::int64_t>(n) * n * n;
    double lambda[3], lambda_total = 0;
    int sign[3];
    for (int axis = 0; axis < 3; ++axis) {
        lambda[axis] = std::abs(direction[axis]) * factor;
        lambda_total += lambda[axis];
        sign[axis] = direction[axis] >= 0 ? 1 : -1;
    }
    if (lambda_total > 0.5 * (1 + 1e-13)) return false;
    for (std::int64_t cell = 0; cell < cells; ++cell) {
        // Each term is nonnegative: |slope| <= 2*source[cell]. This form
        // avoids subtracting two almost equal full flux divergences in voids.
        double value = (1 - 2 * lambda_total) * source[cell];
        for (int axis = 0; axis < 3; ++axis) {
            if (lambda[axis] == 0) continue;
            const auto left = neighbor(cell, axis, -1, n);
            const auto right = neighbor(cell, axis, 1, n);
            const double slope = limited_slope(source[left], source[cell], source[right]);
            const auto upwind = sign[axis] > 0 ? left : right;
            const auto upwind_left = neighbor(upwind, axis, -1, n);
            const auto upwind_right = neighbor(upwind, axis, 1, n);
            const double upwind_slope = limited_slope(
                source[upwind_left], source[upwind], source[upwind_right]);
            const double own = source[cell] - 0.5 * sign[axis] * slope;
            const double incoming = source[upwind] + 0.5 * sign[axis] * upwind_slope;
            value += lambda[axis] * (own + incoming);
        }
        if (!std::isfinite(value) || value < 0) return false;
        destination[cell] = value;
    }
    return true;
}

bool stream(double* photons, int n, int nd, const double* directions,
            double factor, int threads) {
    const std::int64_t cells = static_cast<std::int64_t>(n) * n * n;
    int invalid = 0;
    // Allocate before entering OpenMP so allocation failure is caught by the
    // outer C-interface exception handler, rather than crossing an OMP region.
    std::vector<double> scratch(static_cast<std::size_t>(threads) * 2 * cells);
    // Scratch memory scales with worker count, not the number of ordinates.
    #pragma omp parallel num_threads(threads) reduction(|:invalid)
    {
        double* stage = scratch.data() + static_cast<std::size_t>(omp_get_thread_num()) * 2 * cells;
        double* next = stage + cells;
        #pragma omp for schedule(static)
        for (int d = 0; d < nd; ++d) {
            double* state = photons + d * cells;
            if (!forward_euler(state, stage, n, directions + 3 * d, factor)
                    || !forward_euler(stage, next, n, directions + 3 * d, factor)) {
                invalid = 1;
                continue;
            }
            for (std::int64_t cell = 0; cell < cells; ++cell)
                state[cell] = 0.5 * state[cell] + 0.5 * next[cell];
        }
    }
    return invalid == 0;
}

bool absorb(const double* capacity, double* ionized, double* photons,
            std::int64_t cells, int directions, double coefficient_dt, int threads) {
    int invalid = 0;
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(|:invalid)
    for (std::int64_t cell = 0; cell < cells; ++cell) {
        const double neutral_difference = capacity[cell] - ionized[cell];
        double radiation = 0;
        for (int d = 0; d < directions; ++d) radiation += photons[d * cells + cell];
        if (!std::isfinite(radiation)
                || neutral_difference < -ion_bound_tolerance * capacity[cell]) {
            invalid = 1;
            continue;
        }
        // Preserve the ion count, including permitted accumulated roundoff.
        // A tiny excess creates no negative opacity and absorbs no photons.
        const double neutral = std::max(neutral_difference, 0.);
        if (radiation <= 0 || neutral == 0 || coefficient_dt == 0) continue;
        const double smaller = std::min(radiation, neutral);
        const double difference = std::abs(radiation - neutral);
        const double argument = coefficient_dt * difference;
        double absorbed;
        if (argument > 700) {
            absorbed = smaller;
        } else {
            const double one_minus = -std::expm1(-argument);
            const double phi = argument == 0 ? 1 : one_minus / argument;
            const double product = smaller * coefficient_dt * phi;
            absorbed = std::isinf(product) ? smaller
                : smaller * (one_minus + product) / (1 + product);
        }
        if (!std::isfinite(absorbed) || absorbed < 0 || absorbed > smaller * (1 + 1e-13)) {
            invalid = 1;
            continue;
        }
        absorbed = std::min(absorbed, smaller);
        const double fraction = absorbed / radiation;
        for (int d = 0; d < directions; ++d) {
            double& value = photons[d * cells + cell];
            value = std::fma(-fraction, value, value);
        }
        ionized[cell] += absorbed;
    }
    return invalid == 0;
}

void inject(double* photons, const double* first, const double* last,
            const double* weights, std::int64_t cells, int directions,
            double duration, double first_fraction, double last_fraction, int threads) {
    const double mean = 0.5 * (first_fraction + last_fraction);
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int d = 0; d < directions; ++d) {
        const double factor = duration * weights[d];
        for (std::int64_t cell = 0; cell < cells; ++cell)
            photons[d * cells + cell] += factor * ((1 - mean) * first[cell] + mean * last[cell]);
    }
}
}

extern "C" int causal_rt_evolve(
    int n, int nd, int threads, const double* capacity, double* ionized, double* photons,
    const double* rate_start, const double* rate_end, const double* vectors,
    const double* weights, double duration, double z_start, double z_end,
    double dx_cm, double hydrogen_density, double sigma, double cfl,
    std::int64_t maximum_steps, std::int64_t* used_steps) {
    try {
        const std::int64_t cells = static_cast<std::int64_t>(n) * n * n;
        double angular_max = 0;
        for (int d = 0; d < nd; ++d)
            angular_max = std::max(angular_max, std::abs(vectors[3*d])
                + std::abs(vectors[3*d+1]) + std::abs(vectors[3*d+2]));
        double time = 0;
        *used_steps = 0;
        while (time < duration) {
            if (++*used_steps > maximum_steps) return 1;
            const double z_left = z_start + (z_end - z_start) * time / duration;
            const double stable_dt = cfl * (1 - 8 * std::numeric_limits<double>::epsilon())
                * dx_cm / (light_speed * (1 + z_left) * angular_max);
            const double dt = std::min(stable_dt, duration - time);
            if (!std::isfinite(dt) || dt <= 0 || time + dt == time) return 2;
            const double midpoint = time + 0.5 * dt;
            const double z_mid = z_start + (z_end - z_start) * midpoint / duration;
            const double scale = 1 + z_mid;
            const double reaction_half = 0.5 * dt * light_speed * sigma * hydrogen_density
                * scale * scale * scale;
            inject(photons, rate_start, rate_end, weights, cells, nd, 0.5 * dt,
                   time / duration, midpoint / duration, threads);
            if (!absorb(capacity, ionized, photons, cells, nd, reaction_half, threads)) return 3;
            if (!stream(photons, n, nd, vectors, light_speed * scale * dt / dx_cm, threads)) return 4;
            if (!absorb(capacity, ionized, photons, cells, nd, reaction_half, threads)) return 3;
            inject(photons, rate_start, rate_end, weights, cells, nd, 0.5 * dt,
                   midpoint / duration, (time + dt) / duration, threads);
            time += dt;
        }
        return 0;
    } catch (const std::exception&) {
        return 5;
    }
}
