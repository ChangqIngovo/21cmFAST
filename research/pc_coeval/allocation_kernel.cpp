// Synchronous source-anchored radial allocation on a periodic cubic lattice.
// This is a frozen-density, infinite-speed allocation model, not angular RT.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>
#include <omp.h>

namespace {
struct Offset { int x, y, z, radius2; };
struct Shell { std::size_t begin, end; };

// Reserve small chunks to avoid a contended atomic operation at every edge.
// A reservation failure is a conservative work-limit stop; no partial round
// is committed. Unused reservations are returned before reporting work used.
class VisitBudget {
    std::atomic<std::uint64_t>& reserved;
    const std::uint64_t limit;
    std::uint64_t available = 0;
public:
    VisitBudget(std::atomic<std::uint64_t>& counter, std::uint64_t maximum)
        : reserved(counter), limit(maximum) {}
    ~VisitBudget() { if (available) reserved.fetch_sub(available); }
    bool visit() {
        if (!available) {
            auto before = reserved.load(std::memory_order_relaxed);
            for (;;) {
                if (before >= limit) return false;
                const auto chunk = std::min<std::uint64_t>(4096, limit-before);
                if (reserved.compare_exchange_weak(before, before+chunk)) {
                    available = chunk;
                    break;
                }
            }
        }
        --available;
        return true;
    }
};

inline std::size_t target(int x, int y, int z, const Offset& d, int n) {
    int a=x+d.x, b=y+d.y, c=z+d.z;
    if (a<0) a+=n; else if (a>=n) a-=n;
    if (b<0) b+=n; else if (b>=n) b-=n;
    if (c<0) c+=n; else if (c>=n) c-=n;
    return (static_cast<std::size_t>(a)*n+b)*n+c;
}

long double total(const double* p, std::size_t size) {
    long double sum=0;
    for (std::size_t i=0; i<size; ++i) sum+=p[i];
    return sum;
}

void geometry(int n, std::vector<Offset>& offsets, std::vector<Shell>& shells) {
    offsets.reserve(static_cast<std::size_t>(n)*n*n);
    for (int i=0; i<n; ++i) for (int j=0; j<n; ++j) for (int k=0; k<n; ++k) {
        const int x=i<=n/2 ? i : i-n;
        const int y=j<=n/2 ? j : j-n;
        const int z=k<=n/2 ? k : k-n;
        offsets.push_back({x,y,z,x*x+y*y+z*z});
    }
    std::sort(offsets.begin(), offsets.end(), [](const Offset& a, const Offset& b) {
        if (a.radius2!=b.radius2) return a.radius2<b.radius2;
        if (a.x!=b.x) return a.x<b.x;
        if (a.y!=b.y) return a.y<b.y;
        return a.z<b.z;
    });
    for (std::size_t begin=0; begin<offsets.size();) {
        std::size_t end=begin+1;
        while (end<offsets.size() && offsets[end].radius2==offsets[begin].radius2) ++end;
        shells.push_back({begin,end});
        begin=end;
    }
}
}

// Return 0: complete; 1: iteration limit; 2: work limit; 3: numerical error.
// stats: photons, atoms, ions, remaining, excess, unresolved, relative ledger,
//        kernel seconds, maximum source overdraft, geometry seconds.
// counts: committed rounds, source-target visits, peak active sources,
//         shell count, lattice offsets, closure shortcut (0/1).
extern "C" int allocate_coeval(
    int n, const double* source, const double* capacity, int threads,
    double tolerance, int max_iterations, std::uint64_t max_interactions,
    double* ionized, double* remaining, double* stats, std::uint64_t* counts) {
    const auto started=std::chrono::steady_clock::now();
    try {
        const std::size_t size=static_cast<std::size_t>(n)*n*n;
        const long double photons=total(source,size), atoms=total(capacity,size);
        const long double excess=std::max(0.L,photons-atoms);
        const long double threshold=tolerance*photons;
        std::vector<double> space(size);
        for (std::size_t i=0; i<size; ++i) {
            ionized[i]=std::min(source[i],capacity[i]);
            remaining[i]=source[i]-ionized[i];
            space[i]=capacity[i]-ionized[i];
        }
        std::fill(counts,counts+6,0);
        double maximum_overdraft=0, geometry_seconds=0;
        int result=0;
        std::atomic<std::uint64_t> visits(0);
        if (photons>=atoms && photons>0) {
            // With full periodic reachability the final field is unique.
            // Keep surplus at sources without ever overdrawing their budgets.
            const long double left=total(remaining,size);
            const long double fraction=left>0 ? excess/left : 0;
            for (std::size_t i=0; i<size; ++i) {
                remaining[i]=static_cast<double>(remaining[i]*fraction);
                ionized[i]=capacity[i];
            }
            counts[5]=1;
        } else if (total(remaining,size)>threshold) {
            std::vector<Offset> offsets;
            std::vector<Shell> shells;
            const auto geometry_started=std::chrono::steady_clock::now();
            geometry(n,offsets,shells);
            geometry_seconds=std::chrono::duration<double>(
                std::chrono::steady_clock::now()-geometry_started).count();
            counts[3]=shells.size(); counts[4]=offsets.size();
            std::vector<double> claims(size), acceptance(size);
            std::vector<std::size_t> first_shell(size,0);
            std::vector<std::size_t> active, boundary;
            std::vector<double> fraction, debit;
            for (int round=0; round<max_iterations; ++round) {
                if (total(remaining,size)<=threshold) break;
                active.clear();
                for (std::size_t i=0; i<size; ++i) if (remaining[i]>0) active.push_back(i);
                counts[2]=std::max<std::uint64_t>(counts[2],active.size());
                boundary.assign(active.size(),0);
                fraction.assign(active.size(),0);
                debit.assign(active.size(),0);
                std::fill(claims.begin(),claims.end(),0);
                std::atomic<bool> limited(false), invalid(false);
                const int workers=std::min<int>(threads,static_cast<int>(active.size()));
                #pragma omp parallel num_threads(workers)
                {
                    VisitBudget work(visits,max_interactions);
                    #pragma omp for schedule(dynamic,8)
                    for (std::size_t s=0; s<active.size(); ++s) {
                        if (limited.load()) continue;
                        const auto id=active[s];
                        const int x=id/(n*n), y=(id/n)%n, z=id%n;
                        long double left=remaining[id];
                        for (std::size_t h=first_shell[id]; h<shells.size(); ++h) {
                            const auto shell=shells[h];
                            long double demand=0;
                            for (auto e=shell.begin; e<shell.end; ++e) {
                                if (!work.visit()) { limited=true; break; }
                                demand+=space[target(x,y,z,offsets[e],n)];
                            }
                            if (limited.load()) break;
                            if (demand<=0) continue;
                            const bool partial=left<demand;
                            // Bias only the last floating point quotient downward.
                            const double share=partial ? std::nextafter(
                                static_cast<double>(left/demand),0.0) : 1.0;
                            for (auto e=shell.begin; e<shell.end; ++e) {
                                if (!work.visit()) { limited=true; break; }
                                const auto j=target(x,y,z,offsets[e],n);
                                const double proposal=space[j]*share;
                                #pragma omp atomic update
                                claims[j]+=proposal;
                            }
                            boundary[s]=h;
                            fraction[s]=share;
                            if (limited.load() || partial || left==demand) break;
                            left-=demand;
                        }
                    }
                }
                if (limited.load()) { result=2; break; }
                #pragma omp parallel for num_threads(threads) schedule(static)
                for (std::size_t j=0; j<size; ++j) {
                    acceptance[j]=claims[j]>space[j] ? space[j]/claims[j] : 1.0;
                    if (!std::isfinite(claims[j]) || claims[j]<0) invalid=true;
                }
                if (invalid.load()) { result=3; break; }
                #pragma omp parallel num_threads(workers)
                {
                    VisitBudget work(visits,max_interactions);
                    #pragma omp for schedule(dynamic,8)
                    for (std::size_t s=0; s<active.size(); ++s) {
                        if (limited.load()) continue;
                        const auto id=active[s];
                        const int x=id/(n*n), y=(id/n)%n, z=id%n;
                        long double accepted=0;
                        for (std::size_t h=first_shell[id]; h<=boundary[s]; ++h) {
                            const auto shell=shells[h];
                            const double share=h==boundary[s] ? fraction[s] : 1.0;
                            for (auto e=shell.begin; e<shell.end; ++e) {
                                if (!work.visit()) { limited=true; break; }
                                const auto j=target(x,y,z,offsets[e],n);
                                accepted+=static_cast<long double>(space[j]*share)*acceptance[j];
                            }
                            if (limited.load()) break;
                        }
                        debit[s]=static_cast<double>(accepted);
                        const double allowance=128*std::numeric_limits<double>::epsilon()
                            *std::max(source[id],std::numeric_limits<double>::min());
                        if (!std::isfinite(debit[s]) || debit[s]<0
                            || debit[s]>remaining[id]+allowance) invalid=true;
                    }
                }
                if (limited.load()) { result=2; break; }
                if (invalid.load()) { result=3; break; }
                // Commit an entire synchronous round only after both passes finish.
                for (std::size_t s=0; s<active.size(); ++s) {
                    const auto id=active[s];
                    maximum_overdraft=std::max(maximum_overdraft,debit[s]-remaining[id]);
                    remaining[id]=std::max(0.0,remaining[id]-debit[s]);
                    // All inner shells received this source's full-capacity
                    // claim, hence were exhausted globally in this round.
                    // Keep the partial boundary; never rescan exhausted cores.
                    first_shell[id]=boundary[s]+(fraction[s]==1.0 ? 1 : 0);
                }
                #pragma omp parallel for num_threads(threads) schedule(static)
                for (std::size_t j=0; j<size; ++j) {
                    const double taken=std::min(space[j],claims[j]);
                    space[j]-=taken;
                    // Derive occupied capacity from its bounded remainder.
                    // Repeated additions can overshoot capacity by one ULP.
                    // Untouched cells keep their original local absorption.
                    if (taken>0) ionized[j]=capacity[j]-space[j];
                }
                ++counts[0];
            }
            if (result==0 && total(remaining,size)>threshold) result=1;
        }
        const long double ions=total(ionized,size), left=total(remaining,size);
        stats[0]=static_cast<double>(photons); stats[1]=static_cast<double>(atoms);
        stats[2]=static_cast<double>(ions); stats[3]=static_cast<double>(left);
        stats[4]=static_cast<double>(excess);
        stats[5]=static_cast<double>(std::max(0.L,left-excess));
        stats[6]=photons>0 ? static_cast<double>((photons-ions-left)/photons) : 0;
        stats[7]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        stats[8]=maximum_overdraft; stats[9]=geometry_seconds;
        counts[1]=visits.load();
        return result;
    } catch (...) { return 3; }
}
