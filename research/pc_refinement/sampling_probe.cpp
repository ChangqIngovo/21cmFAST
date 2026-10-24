#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

// Initial-geometry predictor only. The origin is transparent because source
// rays materialize only after that source cell clears in the event solver.
extern "C" int probe_distances(int n, const double *remaining, int ns,
    const int32_t *sources, int nk, const double *directions, double horizon,
    int periodic, double *distances) {
    for (int is=0; is<ns; ++is) {
        int source=sources[is];
        long double origin[3]={(long double)(source/(n*n))+.5L,
            (long double)((source/n)%n)+.5L,(long double)(source%n)+.5L};
        for(int k=0;k<nk;++k) {
            const double *v=directions+3*k;
            int voxel[3]={source/(n*n),(source/n)%n,source%n};
            long double travelled=0;
            double answer=horizon;
            for(int loop=0;loop<1000000;++loop) {
                bool outside=false;
                int ijk[3];
                for(int a=0;a<3;++a) {
                    outside |= voxel[a]<0 || voxel[a]>=n;
                    ijk[a]=(voxel[a]%n+n)%n;
                }
                if((outside&&!periodic)||travelled>=horizon)break;
                int cell=(ijk[0]*n+ijk[1])*n+ijk[2];
                if(cell!=source && remaining[cell]>0) {
                    answer=(double)travelled;break;
                }
                long double crossings[3],next=std::numeric_limits<long double>::infinity();
                for(int a=0;a<3;++a) {
                    crossings[a]=std::numeric_limits<long double>::infinity();
                    if(v[a]!=0) {
                        long double face=(long double)voxel[a]+(v[a]>0?1:0);
                        crossings[a]=(face-origin[a])/(long double)v[a];
                        next=std::min(next,crossings[a]);
                    }
                }
                if(!std::isfinite(next)||next<travelled)return 1;
                if(next>=horizon)break;
                for(int a=0;a<3;++a)if(crossings[a]==next)voxel[a]+=v[a]>0?1:-1;
                travelled=next;
                if(loop==999999)return 2;
            }
            distances[is*nk+k]=answer;
        }
    }
    return 0;
}
