// Exact straight-line first exit from a periodic piecewise-constant voxel phase.
// This is a geometric morphology statistic, not absorption radiative transfer.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

extern "C" int trace_mfp(int n, const std::uint8_t* phase, std::int64_t rays,
                         const double* origins, const double* directions,
                         double cell_size, double cap, int threads,
                         double* distance, std::uint8_t* censored) {
    const double inf = std::numeric_limits<double>::infinity();
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (std::int64_t r=0; r<rays; ++r) {
        int cell[3], step[3];
        double next[3], delta[3];
        for (int a=0; a<3; ++a) {
            const double p = origins[3*r+a]/cell_size;
            cell[a] = std::min(n-1, static_cast<int>(std::floor(p)));
            const double v = directions[3*r+a];
            step[a] = (v>0) - (v<0);
            if (v>0) next[a] = ((cell[a]+1)*cell_size-origins[3*r+a])/v;
            else if (v<0) next[a] = (cell[a]*cell_size-origins[3*r+a])/v;
            else next[a] = inf;
            delta[a] = v==0 ? inf : cell_size/std::abs(v);
        }
        distance[r] = cap;
        censored[r] = 1;
        while (true) {
            const double t = std::min(next[0], std::min(next[1], next[2]));
            if (t>=cap) break;
            // Cross every face at a simultaneous edge/corner intersection.
            const double tie = 16*std::numeric_limits<double>::epsilon()*std::max(1., std::abs(t));
            for (int a=0; a<3; ++a) {
                if (next[a]<=t+tie) {
                    cell[a] += step[a];
                    if (cell[a]<0) cell[a]+=n;
                    else if (cell[a]>=n) cell[a]-=n;
                    next[a] += delta[a];
                }
            }
            const auto index = (static_cast<std::int64_t>(cell[0])*n+cell[1])*n+cell[2];
            if (!phase[index]) {
                distance[r] = std::max(0., t);
                censored[r] = 0;
                break;
            }
        }
    }
    return 0;
}
