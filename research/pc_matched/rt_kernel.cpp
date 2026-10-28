// Direct finite-volume photon-number radiative transfer, periodic boundaries.
// Radiation is carried between intervals; atoms absorb through finite opacity.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <vector>

namespace {
constexpr double light_speed = 2.99792458e10;

bool absorb(const double* capacity, double* ionized, double* photons,
            std::int64_t cells, int directions, double coefficient_dt, int threads) {
    int invalid = 0;
    // Each cell retains its serial directional summation order. Only the
    // integer failure flag is reduced across threads, never physical counts.
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(|:invalid)
    for (std::int64_t cell = 0; cell < cells; ++cell) {
        const double neutral = capacity[cell] - ionized[cell];
        double radiation = 0.0;
        for (int d = 0; d < directions; ++d) radiation += photons[d * cells + cell];
        if (!std::isfinite(radiation) || neutral < -1e-13 * capacity[cell]) {
            invalid = 1;
            continue;
        }
        if (radiation <= 0 || neutral <= 0 || coefficient_dt == 0) continue;
        const double smaller = std::min(radiation, neutral);
        const double difference = std::abs(radiation - neutral);
        const double argument = coefficient_dt * difference;
        double absorbed;
        if (argument > 700) {
            absorbed = smaller;
        } else {
            const double one_minus = -std::expm1(-argument);
            const double phi = argument == 0 ? 1.0 : one_minus / argument;
            const double product = smaller * coefficient_dt * phi;
            absorbed = std::isinf(product) ? smaller : smaller * (one_minus + product) / (1 + product);
        }
        if (!std::isfinite(absorbed) || absorbed < 0 || absorbed > smaller * (1 + 1e-13)) {
            invalid = 1;
            continue;
        }
        // The exact expression lies in [0, smaller]. Restore only a possible
        // final-operation upward roundoff; larger violations were rejected.
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

void inject(double* photons, const double* start, const double* end,
            const double* weights, std::int64_t cells, int directions,
            double interval, double relative_start, double relative_end, int threads) {
    const double mean_fraction = 0.5 * (relative_start + relative_end);
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int d = 0; d < directions; ++d) {
        const double factor = interval * weights[d];
        for (std::int64_t cell = 0; cell < cells; ++cell) {
            const double rate = (1 - mean_fraction) * start[cell] + mean_fraction * end[cell];
            photons[d * cells + cell] += factor * rate;
        }
    }
}
}

extern "C" int rt_evolve(
    int n, int nd, int threads, const double* capacity, double* ionized, double* photons,
    const double* rate_start, const double* rate_end, const double* vectors,
    const double* weights, double duration, double z_start, double z_end,
    double dx_cm, double hydrogen_density, double sigma, double cfl,
    std::int64_t maximum_steps, std::int64_t* used_steps) {
    try {
        const std::int64_t cells = static_cast<std::int64_t>(n) * n * n;
        std::vector<double> temporary(static_cast<std::size_t>(nd) * cells);
        double angular_max = 0;
        for (int d = 0; d < nd; ++d) {
            angular_max = std::max(angular_max, std::abs(vectors[3*d]) +
                std::abs(vectors[3*d+1]) + std::abs(vectors[3*d+2]));
        }
        double time = 0;
        *used_steps = 0;
        while (time < duration) {
            if (++*used_steps > maximum_steps) return 1;
            const double z_left = z_start + (z_end - z_start) * time / duration;
            const double stable_dt = cfl * dx_cm / (light_speed * (1 + z_left) * angular_max);
            const double dt = std::min(stable_dt, duration - time);
            if (!std::isfinite(dt) || dt <= 0 || time + dt == time) return 2;
            const double midpoint = time + 0.5 * dt;
            const double z_mid = z_start + (z_end - z_start) * midpoint / duration;
            const double scale = 1 + z_mid;
            const double reaction_half = 0.5 * dt * light_speed * sigma * hydrogen_density * scale * scale * scale;
            // Symmetric source / reaction / transport / reaction / source split.
            // Upwind transport is first order; source integrals are exact for
            // the supplied linearly interpolated interval emissivity.
            inject(photons, rate_start, rate_end, weights, cells, nd, 0.5 * dt,
                   time / duration, midpoint / duration, threads);
            if (!absorb(capacity, ionized, photons, cells, nd, reaction_half, threads)) return 3;
            const double transport_factor = light_speed * scale * dt / dx_cm;
            int invalid_flux = 0;
            #pragma omp parallel for num_threads(threads) schedule(static) reduction(|:invalid_flux)
            for (int d = 0; d < nd; ++d) {
                const double vx = vectors[3*d];
                const double vy = vectors[3*d+1];
                const double vz = vectors[3*d+2];
                const double nx = std::abs(vx) * transport_factor;
                const double ny = std::abs(vy) * transport_factor;
                const double nz = std::abs(vz) * transport_factor;
                const double stay = 1 - nx - ny - nz;
                if (stay < -1e-14) {
                    invalid_flux = 1;
                    continue;
                }
                const double* src = photons + d * cells;
                double* dest = temporary.data() + d * cells;
                for (int x = 0; x < n; ++x) {
                    const int from_x = (x + (vx >= 0 ? n - 1 : 1)) % n;
                    for (int y = 0; y < n; ++y) {
                        const int from_y = (y + (vy >= 0 ? n - 1 : 1)) % n;
                        for (int z = 0; z < n; ++z) {
                            const int from_z = (z + (vz >= 0 ? n - 1 : 1)) % n;
                            const std::int64_t id = (static_cast<std::int64_t>(x) * n + y) * n + z;
                            dest[id] = stay * src[id]
                                + nx * src[(static_cast<std::int64_t>(from_x) * n + y) * n + z]
                                + ny * src[(static_cast<std::int64_t>(x) * n + from_y) * n + z]
                                + nz * src[(static_cast<std::int64_t>(x) * n + y) * n + from_z];
                        }
                    }
                }
            }
            if (invalid_flux) return 4;
            std::copy(temporary.begin(), temporary.end(), photons);
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
