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

// Positive moments of a Gaussian on [0,1]. Long-double intermediates are
// essential when a tiny stellar source irradiates a mostly neutral cell.
struct Moments {
    long double zeroth;
    long double first;
};

Moments gaussian_quadrature(long double a, long double b, long double centre = 0) {
    // Sixteen-point Gauss-Legendre, used only when the exponent varies by <=4.
    // The remaining (possibly extremely stiff) cases use analytic moments.
    constexpr long double x[8] = {
        .095012509837637440185L, .281603550779258913230L,
        .458016777657227386342L, .617876244402643748447L,
        .755404408355003033895L, .865631202387831743880L,
        .944575023073232576078L, .989400934991649932596L};
    constexpr long double w[8] = {
        .189450610455068496285L, .182603415044923588867L,
        .169156519395002538189L, .149595988816576732081L,
        .124628971255533872052L, .095158511682492784810L,
        .062253523938647892863L, .027152459411754094852L};
    Moments result{0, 0};
    for (int i = 0; i < 8; ++i)
        for (int sign : {-1, 1}) {
            const long double u = .5L * (1 + sign * x[i]);
            const long double delta = u - centre;
            const long double value = .5L * w[i] * std::exp(-a * u - b * delta * delta);
            result.zeroth += value;
            result.first += u * value;
        }
    return result;
}

Moments infinite_decreasing_moments(long double a, long double b) {
    // Integrals of exp(-a*u-b*u*u) and u*exp(-a*u-b*u*u), from 0 to infinity.
    // a,b>=0 and at least one is positive. Large a/sqrt(b) needs the moment
    // expansions: subtracting 1-a*J would otherwise lose the first moment.
    if (b == 0) return {1 / a, 1 / (a * a)};
    const long double root_b = std::sqrt(b);
    if (a >= 24 * root_b) {
        const long double r = b / (a * a);
        long double sum0 = 1, sum1 = 1, term0 = 1, term1 = 1;
        const long double epsilon = std::numeric_limits<long double>::epsilon();
        for (int j = 1; j < 64; ++j) {
            term0 *= -2 * r * (2 * j - 1);
            term1 *= -2 * r * (2 * j + 1);
            sum0 += term0;
            sum1 += term1;
            if (std::abs(term0) <= epsilon * std::abs(sum0)
                    && std::abs(term1) <= epsilon * std::abs(sum1)) break;
        }
        return {sum0 / a, sum1 / (a * a)};
    }
    constexpr long double sqrt_pi = 1.7724538509055160272981674833411451828L;
    const long double x = a / (2 * root_b);
    const long double erfcx = std::exp(x * x) * std::erfc(x);
    const long double zeroth = sqrt_pi * erfcx / (2 * root_b);
    return {zeroth, (1 - a * zeroth) / (2 * b)};
}

Moments decreasing_moments(long double a, long double b) {
    // On [0,1], exp(-a*u-b*u*u) decreases from one. The finite upper-limit
    // correction is safe here: its exponent exceeds four in this branch.
    if (a + b <= 4) return gaussian_quadrature(a, b);
    const auto full = infinite_decreasing_moments(a, b);
    const long double tail = std::exp(-a - b);
    if (tail == 0) return full;
    const auto beyond = infinite_decreasing_moments(a + 2 * b, b);
    return {full.zeroth - tail * beyond.zeroth,
            full.first - tail * (beyond.zeroth + beyond.first)};
}

struct ReactionMoments {
    long double inverse_peak;
    long double source_early; // integral (1-s)*F(s)/max(F) ds
    long double source_late;  // integral s*F(s)/max(F) ds
};

ReactionMoments reaction_moments(long double difference, long double emitted,
                                long double coefficient) {
    // F(s)=exp[-coefficient*(difference*s+emitted*s*s/2)]. Dividing by
    // its maximum prevents overflow, including when the neutral capacity is
    // much larger than the radiation. Both first moments stay nonnegative.
    const long double b = .5L * coefficient * emitted;
    if (difference >= 0) {
        const auto m = decreasing_moments(coefficient * difference, b);
        return {1, m.zeroth - m.first, m.first};
    }
    if (difference + emitted <= 0) {
        const auto m = decreasing_moments(-coefficient * (difference + emitted), b);
        const long double peak = -coefficient * (difference + .5L * emitted);
        return {std::exp(-peak), m.first, m.zeroth - m.first};
    }
    const long double centre = -difference / emitted;
    const long double peak = b * centre * centre;
    long double zeroth, early, late;
    if (b <= 4) {
        const auto m = gaussian_quadrature(0, b, centre);
        zeroth = m.zeroth;
        late = m.first;
        early = zeroth - late;
    } else {
        constexpr long double sqrt_pi = 1.7724538509055160272981674833411451828L;
        const long double root_b = std::sqrt(b);
        const long double left = centre, right = 1 - centre;
        zeroth = sqrt_pi * (std::erf(root_b * left) + std::erf(root_b * right))
            / (2 * root_b);
        // Compute the smaller moment with a sum of positive terms, then the
        // larger by subtraction. This also handles a peak close to an edge.
        if (left <= right) {
            const long double difference_exp = std::exp(-b * left * left)
                * (-std::expm1(-b * (right * right - left * left)));
            late = left * zeroth + difference_exp / (2 * b);
            early = zeroth - late;
        } else {
            const long double difference_exp = std::exp(-b * right * right)
                * (-std::expm1(-b * (left * left - right * right)));
            early = right * zeroth + difference_exp / (2 * b);
            late = zeroth - early;
        }
    }
    return {std::exp(-peak), early, late};
}

bool source_absorb(const double* capacity, double* ionized, double* photons,
                   const double* first, const double* last, const double* weights,
                   std::int64_t cells, int directions, double duration,
                   double first_fraction, double last_fraction,
                   double coefficient_dt, int threads) {
    const long double mean = .5L * (first_fraction + last_fraction);
    int invalid = 0;
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(|:invalid)
    for (std::int64_t cell = 0; cell < cells; ++cell) {
        const double neutral_difference = capacity[cell] - ionized[cell];
        long double radiation = 0;
        for (int d = 0; d < directions; ++d) radiation += photons[d * cells + cell];
        const long double emitted = static_cast<long double>(duration)
            * ((1 - mean) * first[cell] + mean * last[cell]);
        if (!std::isfinite(radiation) || !std::isfinite(emitted) || emitted < 0
                || neutral_difference < -ion_bound_tolerance * capacity[cell]) {
            invalid = 1;
            continue;
        }
        // Permitted accumulated ion roundoff is preserved, never made into
        // negative opacity or used to change the photon inventory.
        const long double neutral = std::max(neutral_difference, 0.);
        if (neutral == 0 || coefficient_dt == 0) {
            for (int d = 0; d < directions; ++d)
                photons[d * cells + cell] += static_cast<double>(weights[d] * emitted);
            continue;
        }
        long double survival, source_surviving, absorbed;
        if (emitted == 0) {
            // Source-free exact coupled reaction, including equal inventories.
            if (radiation == 0) continue;
            const long double smaller = std::min(radiation, neutral);
            const long double difference = std::abs(radiation - neutral);
            const long double argument = coefficient_dt * difference;
            const long double one_minus = -std::expm1(-argument);
            const long double phi = argument == 0 ? 1 : one_minus / argument;
            const long double product = smaller * coefficient_dt * phi;
            absorbed = std::isinf(product) ? smaller
                : smaller * (one_minus + product) / (1 + product);
            // Direct surviving inventory avoids subtracting two nearly equal
            // photon totals. In opaque cells 1-absorbed/radiation can become
            // negative by one long-double ulp even though the exact tail is
            // positive. This expression also preserves the equal-count limit.
            const long double surviving_smaller = std::isinf(product) ? 0.L
                : smaller * std::exp(-argument) / (1 + product);
            const long double remaining = surviving_smaller
                + std::max(radiation - neutral, 0.L);
            survival = remaining / radiation;
            source_surviving = 0;
        } else {
            const auto m = reaction_moments(radiation - neutral, emitted, coefficient_dt);
            const long double moment = m.source_early + m.source_late;
            const long double optical = coefficient_dt * neutral;
            const long double denominator = m.inverse_peak + optical * moment;
            survival = m.inverse_peak / denominator;
            source_surviving = emitted * (m.inverse_peak + optical * m.source_early) / denominator;
            // This positive expression retains absorption of arbitrarily weak
            // sources. Subtracting the final neutral fraction from its initial
            // value would lose those ion increments to cancellation.
            absorbed = optical * (radiation * moment + emitted * m.source_late) / denominator;
        }
        if (!std::isfinite(absorbed) || !std::isfinite(survival)
                || !std::isfinite(source_surviving) || absorbed < 0
                || absorbed > neutral * (1 + 1e-12L)
                || survival < 0 || survival > 1 || source_surviving < 0
                || source_surviving > emitted * (1 + 1e-12L)) {
            invalid = 1;
            continue;
        }
        for (int d = 0; d < directions; ++d)
            photons[d * cells + cell] = static_cast<double>(
                survival * photons[d * cells + cell] + weights[d] * source_surviving);
        ionized[cell] += static_cast<double>(absorbed);
    }
    return invalid == 0;
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
            if (!source_absorb(capacity, ionized, photons, rate_start, rate_end, weights,
                               cells, nd, 0.5 * dt, time / duration, midpoint / duration,
                               reaction_half, threads)) return 3;
            if (!stream(photons, n, nd, vectors, light_speed * scale * dt / dx_cm, threads)) return 4;
            if (!source_absorb(capacity, ionized, photons, rate_start, rate_end, weights,
                               cells, nd, 0.5 * dt, midpoint / duration, (time + dt) / duration,
                               reaction_half, threads)) return 3;
            time += dt;
        }
        return 0;
    } catch (const std::exception&) {
        return 5;
    }
}
