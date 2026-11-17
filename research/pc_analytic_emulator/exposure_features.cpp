#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Aggregate one feed-forward layer of donor-centred exposure descriptors.
// Each donor contributes fixed input features of the form
// source * max(exposure - latest upstream onset, 0). The layer updates
// a latent onset estimate with a single, bounded piecewise-linear step.
// There is no angular photon inventory or time marching.
extern "C" void time_features(
    int n, int directions, const int* start, const int32_t* offsets,
    const double* weights, const double* source, const double* atoms,
    const double* time, double near_weight, double far_weight,
    double delay, double* times, double* near, double* far, double* escape)
{
    const int count=n*n*n;
    std::vector<int> coordinates(2*n+1);
    for(int x=-n/2; x<n+n/2; ++x) coordinates[x+n/2]=(x+n)%n;
    std::vector<double> rate(count), arrival(count), dose_current(count);
    std::vector<int> xs(count), ys(count), zs(count);
    std::vector<int> active, donor;
    for(int i=0;i<count;++i) {
        xs[i]=i/(n*n)+n/2;
        ys[i]=(i/n)%n+n/2;
        zs[i]=i%n+n/2;
        rate[i]=source[i];
        dose_current[i]=source[i]*time[i];
        if(source[i]>0 && time[i]+delay<1.) active.push_back(i);
    }
    std::fill(near,near+count,0.);
    std::fill(far,far+count,0.);
    *escape=0.;
    for(int d=0; d<directions; ++d) {
        donor=active;
        for(int i:donor) arrival[i]=time[i];
        for(int j=start[d]; j<start[d+1]; ++j) {
            const int* shift=offsets+3*j;
            const bool is_near=j-start[d]<4;
            const double gain=is_near ? near_weight : far_weight;
            double* angular_dose=is_near ? near : far;
            const int sx=shift[0],sy=shift[1],sz=shift[2];
            std::size_t retained=0;
            for(int i:donor) {
                const int receiver=(coordinates[xs[i]+sx]*n+coordinates[ys[i]+sy])*n+coordinates[zs[i]+sz];
                const double start_time=std::max(arrival[i]+delay,0.);
                const double intensity=source[i]*weights[d];
                const double available=std::max(0.,time[receiver]-start_time);
                angular_dose[receiver]+=intensity*(1-start_time);
                if(available>0) {
                    rate[receiver]+=gain*intensity;
                    dose_current[receiver]+=gain*intensity*available;
                }
                const double next=std::max(arrival[i],time[receiver]);
                arrival[i]=next;
                if(next+delay<1.) donor[retained++]=i;
            }
            donor.resize(retained);
            if(donor.empty()) break;
        }
        for(int i:donor) {
            *escape+=source[i]*weights[d]*std::clamp(1.-arrival[i]-delay,0.,1.);
        }
    }
    for(int i=0; i<count; ++i) {
        const double external=near_weight*near[i]+far_weight*far[i];
        if(source[i]+external<atoms[i] || rate[i]<=0.) {
            times[i]=1.;
        } else {
            times[i]=std::clamp(time[i]+(atoms[i]-dose_current[i])/rate[i],0.,1.);
        }
    }
}
