// Conservative positive transfer of species along native LPT mass packets.
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <utility>
#include <vector>

namespace {
using Cloud = std::vector<std::pair<std::int64_t, double>>;

Cloud cloud_weights(const double* p, int n, double width) {
    std::vector<std::pair<int, double>> axis_weights[3];
    for (int a = 0; a < 3; ++a) {
        const double low = p[a] - width/2, high = p[a] + width/2;
        const int first = static_cast<int>(std::floor(low + 0.5));
        const int last = static_cast<int>(std::ceil(high + 0.5)) - 1;
        for (int i = first; i <= last; ++i) {
            const double overlap = std::fmin(high, i+0.5) - std::fmax(low, i-0.5);
            if (overlap <= 0) continue;
            const int index = (i%n+n)%n;
            bool combined = false;
            for (auto &entry : axis_weights[a])
                if (entry.first == index) {
                    entry.second += overlap/width;
                    combined = true;
                    break;
                }
            if (!combined) axis_weights[a].push_back({index, overlap/width});
        }
    }
    Cloud result;
    for (const auto &x : axis_weights[0])
        for (const auto &y : axis_weights[1])
            for (const auto &z : axis_weights[2]) {
                const std::int64_t index = (static_cast<std::int64_t>(x.first)*n+y.first)*n+z.first;
                result.push_back({index, x.second*y.second*z.second});
            }
    std::sort(result.begin(), result.end());
    return result;
}

struct Transfer {
    int from, to;
    double weight;
};

std::vector<Transfer> translated_axis(double old, double next, int n, double width) {
    // Translate material coordinates, not only the cloud's density marginals.
    // The common refinement of old and shifted-new cell boundaries partitions
    // every portion of the cloud into one donor and one receiving cell.
    const double shift = std::remainder(next-old, n);
    const double low = old-width/2, high = old+width/2;
    std::vector<double> edges = {low, high};
    for (int i = static_cast<int>(std::floor(low-0.5))+1; i+0.5 < high; ++i)
        edges.push_back(i+0.5);
    for (int j = static_cast<int>(std::floor(low+shift-0.5))+1; j+0.5-shift < high; ++j) {
        const double edge = j+0.5-shift;
        if (edge > low) edges.push_back(edge);
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    std::vector<Transfer> result;
    for (std::size_t i = 1; i < edges.size(); ++i) {
        const double length = edges[i]-edges[i-1];
        if (length <= 0) continue;
        const double middle = edges[i-1]+length/2;
        int from = static_cast<int>(std::floor(middle+0.5));
        int to = static_cast<int>(std::floor(middle+shift+0.5));
        result.push_back({(from%n+n)%n, (to%n+n)%n, length/width});
    }
    return result;
}
}

extern "C" int gas_deposit(int n, std::int64_t np, double width, const double* mass,
                            const double* positions, double* capacity) {
    try {
        for (std::int64_t p = 0; p < np; ++p) {
            const Cloud cloud = cloud_weights(positions + 3*p, n, width);
            for (const auto &entry : cloud)
                capacity[entry.first] += mass[p] * entry.second;
        }
        return 0;
    } catch (...) { return 1; }
}

extern "C" int gas_remap(int n, std::int64_t np, double width, const double* mass,
                          const double* old_positions, const double* new_positions,
                          const double* fraction, double* ionized) {
    try {
        for (std::int64_t p = 0; p < np; ++p) {
            if (mass[p] == 0) continue;
            const auto tx = translated_axis(old_positions[3*p], new_positions[3*p], n, width);
            const auto ty = translated_axis(old_positions[3*p+1], new_positions[3*p+1], n, width);
            const auto tz = translated_axis(old_positions[3*p+2], new_positions[3*p+2], n, width);
            for (const auto &x : tx)
                for (const auto &y : ty)
                    for (const auto &z : tz) {
                        const std::int64_t from = (static_cast<std::int64_t>(x.from)*n+y.from)*n+z.from;
                        const std::int64_t to = (static_cast<std::int64_t>(x.to)*n+y.to)*n+z.to;
                        ionized[to] += mass[p]*x.weight*y.weight*z.weight*fraction[from];
                    }
        }
        return 0;
    } catch (...) { return 1; }
}
