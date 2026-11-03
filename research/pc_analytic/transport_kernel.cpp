// Analytic-column angular-grid transport, with shared neutral capacity.
// The translation stencil is positive and conservative; its spatial diffusion
// is a numerical approximation, not physical scattering between directions.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <array>
#include <vector>

namespace {
long double absorbed(long double p, long double h, long double k) {
    if (p <= 0 || h <= 0 || k <= 0) return 0;
    const long double small = std::min(p, h);
    const long double a = k * small;
    const long double b = k * std::max(p, h);
    long double value;
    if (a < .5L) {
        value = -std::log1p(-(-std::expm1(-a)) * (-std::expm1(-b))) / k;
    } else {
        value = small - std::log1p(std::exp(-k * std::abs(p-h))
                                  * (-std::expm1(-a))) / k;
    }
    return std::min(small, std::max(0.L, value));
}

void absorb(double* ions, double* radiation, const double* capacity,
            std::int64_t cells, int nd, double k, int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (std::int64_t i=0; i<cells; ++i) {
        long double photons = 0;
        for (int d=0; d<nd; ++d) photons += radiation[d*cells+i];
        const long double neutral = std::max(0., capacity[i]-ions[i]);
        const long double used = absorbed(photons, neutral, k);
        if (used == 0) continue;
        const long double survival = (photons-used)/photons;
        for (int d=0; d<nd; ++d)
            radiation[d*cells+i] = static_cast<double>(radiation[d*cells+i]*survival);
        ions[i] = std::min(capacity[i], static_cast<double>(ions[i]+used));
    }
}

// Exact translation of the piecewise linear interpolation of each angular
// photon grid. Axis translations commute on a periodic mesh. There is no
// transfer between angular bins. The displacement is <= one cell per axis.
void translate(double* radiation, double* temporary, int n, int nd,
               const double* vectors, double distance, bool periodic,
               double* escaped, int threads) {
    const std::int64_t cells = static_cast<std::int64_t>(n)*n*n;
    for (int axis=0; axis<3; ++axis) {
        const std::int64_t stride = axis==0 ? n*n : (axis==1 ? n : 1);
        #pragma omp parallel for num_threads(threads) schedule(static)
        for (int d=0; d<nd; ++d) {
            const double displacement = distance*vectors[3*d+axis];
            const double f = std::abs(displacement);
            const int sign = displacement >= 0 ? 1 : -1;
            long double lost = 0;
            for (std::int64_t i=0; i<cells; ++i) {
                const int coordinate = static_cast<int>((i/stride)%n);
                const int donor_coordinate = coordinate-sign;
                std::int64_t donor = i-sign*stride;
                double upstream = 0;
                if (donor_coordinate>=0 && donor_coordinate<n) {
                    upstream = radiation[d*cells+donor];
                } else if (periodic) {
                    donor += donor_coordinate<0 ? n*stride : -n*stride;
                    upstream = radiation[d*cells+donor];
                }
                const double here = radiation[d*cells+i];
                temporary[d*cells+i] = (1-f)*here+f*upstream;
                if (!periodic && ((sign>0 && coordinate==n-1)
                                  || (sign<0 && coordinate==0))) lost += f*here;
            }
            escaped[d] += static_cast<double>(lost);
        }
        std::copy(temporary, temporary+cells*nd, radiation);
    }
}
}

namespace {
// Simultaneous absorption for unequal full-chord optical depths. In contrast
// to averaging the chords, this integrates the shared H population with the
// actual coefficient of every incident direction. Equal coefficients use the
// exact closed fluence map in the fast path below.
bool unequal_absorption(const std::vector<long double>& p,
                        const std::vector<long double>& k,
                        long double neutral,
                        std::vector<long double>& used) {
    const int count=static_cast<int>(p.size());
    used.assign(count,0.L);
    if (neutral<=0) return true;
    auto rates = [&](long double h, std::vector<long double>& out) {
        out.resize(count);
        for (int d=0; d<count; ++d)
            out[d]=p[d]*(-std::expm1(-k[d]*std::max(0.L,h)));
    };
    auto sum = [&](const std::vector<long double>& values) {
        long double total=0;
        for (auto v:values) total+=v;
        return total;
    };
    auto rk4 = [&](long double h, long double step, std::vector<long double>& delta) {
        std::vector<long double> a,b,c,d;
        rates(h,a);
        const long double h2=h-.5L*step*sum(a);
        if (h2<0) return false;
        rates(h2,b);
        const long double h3=h-.5L*step*sum(b);
        if (h3<0) return false;
        rates(h3,c);
        const long double h4=h-step*sum(c);
        if (h4<0) return false;
        rates(h4,d);
        delta.resize(count);
        for (int q=0; q<count; ++q)
            delta[q]=step*(a[q]+2*b[q]+2*c[q]+d[q])/6;
        return sum(delta)<=h;
    };
    long double photons=sum(p), h=neutral, u=0, step=1;
    const long double scale=std::min(photons,neutral);
    if (scale<=0) return true;
    for (int iteration=0; iteration<20000 && u<1; ++iteration) {
        if (h<=1e-13L*scale) return true; // retain this neutral tail, do not erase it
        std::vector<long double> rate;
        rates(h,rate);
        const long double total=sum(rate);
        if (total<=0) return true;
        step=std::min({step,1-u,.5L*h/total});
        if (step<=0 || u+step==u) return false;
        std::vector<long double> full,first,second;
        if (!rk4(h,step,full) || !rk4(h,.5L*step,first)
                || !rk4(h-sum(first),.5L*step,second)) {
            step*=.5L;
            continue;
        }
        long double error=0;
        for (int q=0; q<count; ++q)
            error+=std::abs(first[q]+second[q]-full[q]);
        // Apportion error to absorbed inventory, not elapsed exposure: a
        // bright source can deplete the atoms in an arbitrarily small du.
        // Positive RK weights ensure each direction consumes <= p_d*du.
        const long double allowed=1e-11L*(sum(first)+sum(second));
        if (error<=allowed) {
            for (int q=0; q<count; ++q) used[q]+=first[q]+second[q];
            h=neutral-sum(used);
            u+=step;
            step*=error>0 ? std::min(2.L,.9L*std::pow(allowed/error,.2L)) : 2.L;
        } else {
            step*=std::max(.1L,.9L*std::pow(allowed/error,.2L));
        }
    }
    return u>=1;
}

bool absorb_chords(double* ions, double* radiation, const double* capacity,
                   std::int64_t cells, const std::vector<int>& active,
                   const std::vector<double>& coefficients, int threads) {
    if (active.empty()) return true;
    bool equal=true;
    for (double value:coefficients)
        equal=equal && std::abs(value-coefficients[0])
            <=1e-12*std::max(value,coefficients[0]);
    int invalid=0;
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(|:invalid)
    for (std::int64_t cell=0; cell<cells; ++cell) {
        const long double neutral=std::max(0.,capacity[cell]-ions[cell]);
        if (neutral<=0) continue;
        long double photons=0;
        for (int d:active) photons+=radiation[d*cells+cell];
        if (photons<=0) continue;
        if (equal) {
            const long double used=absorbed(photons,neutral,coefficients[0]);
            const long double survival=(photons-used)/photons;
            for (int d:active)
                radiation[d*cells+cell]=static_cast<double>(radiation[d*cells+cell]*survival);
            ions[cell]=std::min(capacity[cell],static_cast<double>(ions[cell]+used));
        } else {
            std::vector<long double> p,k,used;
            p.reserve(active.size()); k.reserve(active.size());
            for (std::size_t j=0; j<active.size(); ++j) {
                p.push_back(radiation[active[j]*cells+cell]);
                k.push_back(coefficients[j]);
            }
            if (!unequal_absorption(p,k,neutral,used)) {invalid=1; continue;}
            long double total=0;
            for (std::size_t j=0; j<active.size(); ++j) {
                if (used[j]<0 || used[j]>p[j]*(1+1e-12L)) {invalid=1; continue;}
                radiation[active[j]*cells+cell]=static_cast<double>(std::max(0.L,p[j]-used[j]));
                total+=used[j];
            }
            if (total>neutral*(1+1e-12L)) {invalid=1; continue;}
            ions[cell]=std::min(capacity[cell],static_cast<double>(ions[cell]+total));
        }
    }
    return invalid==0;
}

// A common ray phase is possible because every source starts at a cell
// centre. Each angular field is shifted by an INTEGER voxel displacement at
// its next face crossing. No interpolation or cross-direction transfer occurs.
void roll_direction(double* radiation, double* temporary, int n, int d,
                    const std::array<int,3>& shift, bool periodic,
                    bool terminate, double* escaped) {
    const std::int64_t cells=static_cast<std::int64_t>(n)*n*n;
    double* field=radiation+d*cells;
    std::fill(temporary,temporary+cells,0.);
    long double lost=0;
    for (int i=0;i<n;++i) for (int j=0;j<n;++j) for (int k=0;k<n;++k) {
        const std::int64_t index=(static_cast<std::int64_t>(i)*n+j)*n+k;
        int a=i+shift[0],b=j+shift[1],c=k+shift[2];
        if (terminate || (!periodic && (a<0 || a>=n || b<0 || b>=n || c<0 || c>=n))) {
            lost+=field[index];
        } else {
            a=(a+n)%n; b=(b+n)%n; c=(c+n)%n;
            temporary[(static_cast<std::int64_t>(a)*n+b)*n+c]=field[index];
        }
    }
    escaped[d]+=static_cast<double>(lost);
    std::copy(temporary,temporary+cells,field);
}
}

extern "C" int analytic_shared_cell(int count, const double* photons,
                                    const double* coefficients, double neutral,
                                    double* absorbed_out) {
    try {
        std::vector<long double> p(count),k(count),used;
        for (int d=0;d<count;++d) {p[d]=photons[d];k[d]=coefficients[d];}
        if (!unequal_absorption(p,k,neutral,used)) return 1;
        for (int d=0;d<count;++d) absorbed_out[d]=static_cast<double>(used[d]);
        return 0;
    } catch (...) {return 2;}
}

extern "C" int analytic_characteristics(
    int n, int nd, int threads, const double* angular_source,
    const double* capacity, const double* vectors,
    double cell_coefficient, int exposure_steps, int boundary,
    double tolerance, std::int64_t maximum_events,
    double* ions, double* radiation, double* escaped, double* statistics) {
    try {
        const std::int64_t cells=static_cast<std::int64_t>(n)*n*n;
        const bool periodic=boundary!=1;
        const bool capped=boundary==2;
        const double infinity=std::numeric_limits<double>::infinity();
        std::vector<double> current(cells*nd,0.),temporary(cells*nd,0.);
        long double source_total=0,atom_total=0,initial_ions=0;
        for (std::int64_t i=0;i<cells;++i) {
            atom_total+=capacity[i]; initial_ions+=ions[i];
            for (int d=0;d<nd;++d) source_total+=angular_source[d*cells+i];
        }
        const long double initial_neutral=atom_total-initial_ions;
        const long double normalization=std::max(
            initial_neutral>0 ? std::min(source_total,initial_neutral) : source_total,1e-300L);
        const long double threshold=tolerance*normalization/exposure_steps;
        std::int64_t events=0;
        int injected_batches=0,status=0;
        long double injected=0,maximum_retired=0;
        for (int batch=0;batch<exposure_steps;++batch) {
            const double fraction=batch==exposure_steps-1
                ? 1.-static_cast<double>(batch)/exposure_steps : 1./exposure_steps;
            for (std::int64_t i=0;i<cells*nd;++i) current[i]=angular_source[i]*fraction;
            injected+=source_total*fraction; ++injected_batches;
            std::vector<std::array<double,3>> next(nd);
            std::vector<std::array<int,3>> offset(nd);
            std::vector<double> last(nd,0.);
            std::vector<bool> done(nd,false);
            for (int d=0;d<nd;++d) {
                offset[d]={0,0,0};
                long double photons=0;
                for (std::int64_t i=0;i<cells;++i) photons+=current[d*cells+i];
                done[d]=photons==0;
                for (int a=0;a<3;++a) {
                    const double velocity=std::abs(vectors[3*d+a]);
                    next[d][a]=velocity>1e-15 ? .5/velocity : infinity;
                }
            }
            while (true) {
                long double photons=0,neutral=0;
                for (std::int64_t i=0;i<cells;++i) {
                    neutral+=std::max(0.,capacity[i]-ions[i]);
                    for (int d=0;d<nd;++d) photons+=current[d*cells+i];
                }
                if (photons<=threshold || (boundary==0 && neutral<=threshold)
                        || (boundary==0 && cell_coefficient==0)) break;
                if (events>=maximum_events) {status=1; break;}
                double distance=infinity;
                for (int d=0;d<nd;++d) if (!done[d])
                    for (int a=0;a<3;++a) distance=std::min(distance,next[d][a]);
                if (!std::isfinite(distance)) break;
                const double epsilon=128*std::numeric_limits<double>::epsilon()*std::max(distance,1.);
                std::vector<int> active;
                std::vector<double> coefficients;
                std::vector<std::array<int,3>> shifts;
                std::vector<bool> terminate;
                for (int d=0;d<nd;++d) if (!done[d]) {
                    std::array<int,3> shift{0,0,0};
                    bool touched=false,leaves=false;
                    for (int a=0;a<3;++a) if (next[d][a]<=distance+epsilon) {
                        touched=true;
                        shift[a]=vectors[3*d+a]>0 ? 1 : -1;
                        offset[d][a]+=shift[a];
                        const int lower=-(n/2),upper=n/2-1+n%2;
                        leaves=leaves || offset[d][a]<lower || offset[d][a]>upper;
                        next[d][a]+=1/std::abs(vectors[3*d+a]);
                    }
                    if (touched) {
                        active.push_back(d);
                        coefficients.push_back(cell_coefficient*(distance-last[d]));
                        shifts.push_back(shift);
                        terminate.push_back(capped && leaves);
                        last[d]=distance;
                    }
                }
                if (!absorb_chords(ions,current.data(),capacity,cells,active,coefficients,threads)) {
                    status=3;break;
                }
                #pragma omp parallel for num_threads(threads) schedule(static)
                for (std::size_t j=0;j<active.size();++j) {
                    const int d=active[j];
                    roll_direction(current.data(),temporary.data()+d*cells,n,d,shifts[j],
                                   periodic,terminate[j],escaped);
                }
                for (std::size_t j=0;j<active.size();++j)
                    if (terminate[j]) done[active[j]]=true;
                ++events;
            }
            long double retired=0;
            for (std::int64_t i=0;i<cells*nd;++i) {
                radiation[i]+=current[i];retired+=current[i];
            }
            maximum_retired=std::max(maximum_retired,retired);
            if (status) break;
        }
        statistics[0]=static_cast<double>(events);statistics[1]=injected_batches;
        statistics[2]=static_cast<double>(injected);statistics[3]=static_cast<double>(source_total);
        statistics[4]=static_cast<double>(initial_ions);statistics[5]=static_cast<double>(maximum_retired);
        return status;
    } catch (...) {return 2;}
}

extern "C" int analytic_transport(
    int n, int nd, int threads, const double* angular_source,
    const double* capacity, const double* vectors,
    double coefficient, double distance, int exposure_steps, int periodic,
    double tolerance, double batch_tolerance, std::int64_t maximum_sweeps,
    double* ions, double* radiation, double* escaped, double* statistics) {
    try {
        const std::int64_t cells = static_cast<std::int64_t>(n)*n*n;
        std::vector<double> temporary(cells*nd);
        long double source_total=0, atom_total=0, initial_ions=0;
        for (std::int64_t i=0; i<cells; ++i) {
            atom_total += capacity[i]; initial_ions += ions[i];
            for (int d=0; d<nd; ++d) source_total += angular_source[d*cells+i];
        }
        const long double initial_neutral = atom_total-initial_ions;
        const long double normalization = std::max(
            initial_neutral>0 ? std::min(source_total, initial_neutral) : source_total,
            1e-300L);
        std::int64_t sweeps=0;
        int completed_batches=0;
        long double injected=0;
        long double maximum_carried=0;
        int status=0;
        for (int batch=0; batch<exposure_steps; ++batch) {
            const long double threshold = (batch==exposure_steps-1 ? tolerance : batch_tolerance)
                *normalization/exposure_steps;
            const double fraction = batch==exposure_steps-1
                ? 1.-static_cast<double>(batch)/exposure_steps : 1./exposure_steps;
            #pragma omp parallel for num_threads(threads) schedule(static)
            for (std::int64_t i=0; i<cells*nd; ++i)
                radiation[i] += angular_source[i]*fraction;
            injected += source_total*fraction;
            ++completed_batches;
            while (true) {
                long double photons=0, neutral=0;
                for (std::int64_t i=0; i<cells; ++i) {
                    neutral += std::max(0.,capacity[i]-ions[i]);
                    for (int d=0; d<nd; ++d) photons += radiation[d*cells+i];
                }
                if (photons <= threshold || (periodic && neutral <= threshold)
                    || (periodic && coefficient==0)) {
                    if (batch<exposure_steps-1)
                        maximum_carried=std::max(maximum_carried, std::min(photons,neutral));
                    break;
                }
                if (sweeps>=maximum_sweeps) {status=1; break;}
                // Symmetric splitting avoids assigning a full cell optical
                // path to newly emitted photons before their first move.
                absorb(ions, radiation, capacity, cells, nd, .5*coefficient, threads);
                translate(radiation, temporary.data(), n, nd, vectors,
                          distance, periodic!=0, escaped, threads);
                absorb(ions, radiation, capacity, cells, nd, .5*coefficient, threads);
                ++sweeps;
            }
            if (status) break;
        }
        statistics[0]=static_cast<double>(sweeps);
        statistics[1]=completed_batches;
        statistics[2]=static_cast<double>(injected);
        statistics[3]=static_cast<double>(source_total);
        statistics[4]=static_cast<double>(initial_ions);
        statistics[5]=static_cast<double>(maximum_carried);
        return status;
    } catch (...) {return 2;}
}
