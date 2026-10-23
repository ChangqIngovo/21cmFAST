/* Research-only first-hit characteristics in a FROZEN opacity geometry.
 * No front evolution or recombinations. Exact direction/subcell ray position
 * are retained. The integer clearance is a conservative empty-space bound.
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <omp.h>

int trace_first_hits(int n, const int32_t *clearance,
                     int ns, const double *sources,
                     int nd, const double *directions,
                     double maximum_distance, int use_skips, int threads,
                     int64_t *hits, double *distance, int32_t *visits) {
    if (n < 1 || ns < 1 || nd < 1 || threads < 1 || maximum_distance <= 0)
        return 1;
    const int64_t nrays = (int64_t)ns * nd;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t ray = 0; ray < nrays; ray++) {
        int s = (int)(ray/nd), k = (int)(ray%nd);
        double x[3] = {sources[3*s],sources[3*s+1],sources[3*s+2]};
        const double *v = directions + 3*k;
        double travelled = 0;
        int64_t answer = -1; /* -1 open escape; -2 distance limit; -3 error */
        int count = 0;
        while (++count < 1000000) {
            int ijk[3] = {(int)floor(x[0]),(int)floor(x[1]),(int)floor(x[2])};
            if (ijk[0]<0 || ijk[0]>=n || ijk[1]<0 || ijk[1]>=n || ijk[2]<0 || ijk[2]>=n)
                break;
            int64_t cell = ((int64_t)ijk[0]*n + ijk[1])*n + ijk[2];
            int d = clearance[cell];
            if (d==0) { answer=cell; break; }
            double step;
            if (use_skips && d>=2) {
                /* Any point in this transparent voxel is at least d-1
                 * Euclidean cell lengths from the nearest opaque voxel.
                 * A unit-direction segment shorter than this cannot hit it. */
                step = (double)d - 1.0000000001;
            } else {
                step = INFINITY;
                for (int a=0;a<3;a++) {
                    if (v[a]==0) continue;
                    double face = v[a]>0 ? ijk[a]+1.0 : ijk[a];
                    double t = (face-x[a])/v[a];
                    if (t < step) step=t;
                }
                step += 1e-10; /* disambiguate a boundary point downstream */
            }
            if (!isfinite(step) || step<=0) {answer=-3;break;}
            if (travelled+step > maximum_distance) {
                travelled=maximum_distance; answer=-2;break;
            }
            for(int a=0;a<3;a++) x[a]+=step*v[a];
            travelled+=step;
        }
        if(count>=1000000) answer=-3;
        hits[ray]=answer;distance[ray]=travelled;visits[ray]=count;
    }
    return 0;
}
