"""Positive MC-limited angular streaming with SSPRK2, for research probes.

This improves spatial advection on angular_hybrid's fixed angle grid. It is
still RT, not an ES criterion. Absorption at both RK stages prevents photons
streaming through an unexhausted opaque cell during a full RK step. Ionized
counts are averaged with radiation in the SSP convex combination, preserving
the ledger. No second-order accuracy is asserted across sharp ionization fronts.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import angular_hybrid

_C = r"""
#define PC_ACTIVE_CELLS 1
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static unsigned long long active_updates=0;
unsigned long long get_active_updates(void) {return active_updates;}

static inline double mc(double left, double middle, double right) {
    if(left>0. && middle>0. && right>0.)
        return fmin(2.*left,fmin(middle,2.*right));
    if(left<0. && middle<0. && right<0.)
        return -fmin(-2.*left,fmin(-middle,-2.*right));
    return 0.;
}
static inline double neighbor(const double *p, size_t j, int a, int nd,
        int coordinate, int shift, size_t stride, int n, int periodic) {
    int next=coordinate+shift;
    if(next<0 || next>=n) {
        if(!periodic) return 0.;
        next=(next%n+n)%n;
    }
    size_t other=(size_t)((long long)j+(long long)(next-coordinate)*stride);
    return p[other*nd+a];
}
static void euler(int n, int nd, int threads, int periodic, double dt,
        const double *directions, const double *p, const unsigned char *active,
        double *out, unsigned char *out_active,
        double *escaped) {
    size_t cells=(size_t)n*n*n;
    unsigned long long updates=0;
    #pragma omp parallel for num_threads(threads) reduction(+:escaped[:nd],updates)
    for(size_t j=0;j<cells;++j) {
        int pos[3]={(int)(j/(n*n)),(int)((j/n)%n),(int)(j%n)};
        size_t stride[3]={(size_t)n*n,(size_t)n,1};
        int possible=active[j];
        for(int axis=0;axis<3 && !possible;++axis) {
            for(int shift=-1;shift<=1;shift+=2) {
                int coordinate=pos[axis]+shift;
                if(coordinate<0 || coordinate>=n) {
                    if(!periodic) continue;
                    coordinate=(coordinate+n)%n;
                }
                size_t other=(size_t)((long long)j+(long long)(coordinate-pos[axis])*stride[axis]);
                if(active[other]) possible=1;
            }
        }
        if(PC_ACTIVE_CELLS && !possible) {
            if(out_active[j]) memset(out+j*nd,0,nd*sizeof(double));
            out_active[j]=0;
            continue;
        }
        int any=0;
        updates+=(unsigned long long)nd;
        for(int a=0;a<nd;++a) {
            double value=p[j*nd+a], update=value;
            for(int axis=0;axis<3;++axis) {
                double velocity=directions[3*a+axis];
                double coefficient=fabs(velocity)*dt;
                int sign=velocity>=0 ? 1 : -1;
                double back=neighbor(p,j,a,nd,pos[axis],-sign,stride[axis],n,periodic);
                double ahead=neighbor(p,j,a,nd,pos[axis],sign,stride[axis],n,periodic);
                double back2=neighbor(p,j,a,nd,pos[axis],-2*sign,stride[axis],n,periodic);
                double slope=mc(value-back,.5*(ahead-back),ahead-value);
                double upstream=mc(back-back2,.5*(value-back2),value-back);
                double outgoing=coefficient*(value+.5*slope);
                double incoming=coefficient*(back+.5*upstream);
                update+=incoming-outgoing;
                if(!periodic && ((velocity>0 && pos[axis]==n-1) ||
                                  (velocity<0 && pos[axis]==0)))
                    escaped[a]+=.5*outgoing;
            }
            out[j*nd+a]=update;
            if(update!=0.) any=1;
        }
        out_active[j]=any;
    }
    active_updates+=updates;
}
static void absorb(size_t cells,int nd,int threads,const double *capacity,
        const double *previous,double *next,double *p,unsigned char *active,
        double *by_angle,double weight) {
    #pragma omp parallel for num_threads(threads) reduction(+:by_angle[:nd])
    for(size_t j=0;j<cells;++j) {
        if(PC_ACTIVE_CELLS && !active[j]) {next[j]=previous[j];continue;}
        double total=0.;
        for(int a=0;a<nd;++a) total+=p[j*nd+a];
        double take=fmin(fmax(0.,capacity[j]-previous[j]),total);
        double fraction=total>0 ? take/total : 0.;
        next[j]=previous[j]+take;
        int any=0;
        for(int a=0;a<nd;++a) {
            double delta=p[j*nd+a]*fraction;
            p[j*nd+a]-=delta;
            if(p[j*nd+a]!=0.) any=1;
            by_angle[a]+=weight*delta;
        }
        active[j]=any;
    }
}
int transport(int n,int nd,int steps,int emission_steps,int threads,int periodic,
        double dt,const double *directions,const double *weights,
        const double *source,const double *capacity,double *absorbed,
        double *radiation,double *by_angle,double *escaped) {
    size_t cells=(size_t)n*n*n,elements=cells*nd;
    active_updates=0;
    double *first=calloc(elements,sizeof(double));
    double *second=calloc(elements,sizeof(double));
    double *stage_abs=calloc(cells,sizeof(double));
    unsigned char *active=calloc(cells,1),*first_active=calloc(cells,1),*second_active=calloc(cells,1);
    if(!first || !second || !stage_abs || !active || !first_active || !second_active) {
        free(first);free(second);free(stage_abs);free(active);free(first_active);free(second_active);return 1;
    }
    memset(absorbed,0,cells*sizeof(double));
    memset(radiation,0,elements*sizeof(double));
    memset(by_angle,0,nd*sizeof(double));
    memset(escaped,0,nd*sizeof(double));
    for(int step=0;step<steps;++step) {
        if(step<emission_steps) {
            #pragma omp parallel for num_threads(threads)
            for(size_t j=0;j<cells;++j) {
                if(source[j]==0.) continue;
                active[j]=1;
                for(int a=0;a<nd;++a)
                    radiation[j*nd+a]+=source[j]*weights[a]/emission_steps;
            }
        }
        absorb(cells,nd,threads,capacity,absorbed,absorbed,radiation,active,by_angle,1.);
        euler(n,nd,threads,periodic,dt,directions,radiation,active,first,first_active,escaped);
        absorb(cells,nd,threads,capacity,absorbed,stage_abs,first,first_active,by_angle,.5);
        euler(n,nd,threads,periodic,dt,directions,first,first_active,second,second_active,escaped);
        absorb(cells,nd,threads,capacity,stage_abs,stage_abs,second,second_active,by_angle,.5);
        #pragma omp parallel for num_threads(threads)
        for(size_t j=0;j<cells;++j) {
            absorbed[j]=.5*(absorbed[j]+stage_abs[j]);
            if(!active[j] && !second_active[j]) continue;
            active[j]=1;
            for(int a=0;a<nd;++a)
                radiation[j*nd+a]=.5*(radiation[j*nd+a]+second[j*nd+a]);
        }
    }
    absorb(cells,nd,threads,capacity,absorbed,absorbed,radiation,active,by_angle,1.);
    free(first);free(second);free(stage_abs);
    free(active);free(first_active);free(second_active);
    return 0;
}
"""

_LIB = {}


def _library(active_cells=True):
    if active_cells not in _LIB:
        code = _C if active_cells else _C.replace("#define PC_ACTIVE_CELLS 1", "#define PC_ACTIVE_CELLS 0")
        token = hashlib.sha256(code.encode()).hexdigest()[:16]
        directory = Path(tempfile.gettempdir()) / f"pc_sn_mc_probe_{token}"
        directory.mkdir(exist_ok=True)
        source, binary = directory / "transport.c", directory / "transport.so"
        if not binary.exists():
            source.write_text(code)
            subprocess.run(
                [os.environ.get("CC", "gcc"), "-O3", "-std=c99", "-fopenmp",
                 "-shared", "-fPIC", str(source), "-o", str(binary), "-lm"], check=True,
            )
        library = ctypes.CDLL(str(binary))
        ptr = ctypes.POINTER(ctypes.c_double)
        library.transport.argtypes = [ctypes.c_int]*6+[ctypes.c_double]+[ptr]*8
        library.transport.restype = ctypes.c_int
        library.get_active_updates.restype = ctypes.c_ulonglong
        _LIB[active_cells] = library
    return _LIB[active_cells]


def solve(source, capacity, *, cfl=.45, active_cells=True, **kwargs):
    """Evolve positive second-order streaming; MC positivity requires CFL<=.5."""
    if not 0 < cfl <= .5:
        raise ValueError("MC-limited positivity requires sum-direction CFL <= .5")
    backend = _library(active_cells)
    result = angular_hybrid.solve(source, capacity, cfl=cfl, _backend=backend, **kwargs)
    d = result["diagnostics"]
    d["streaming"] = "MC-limited MUSCL with SSPRK2 and capacity projection at stages"
    d["working_array_bytes"] = int(3*result["radiation"].nbytes+3*result["absorbed"].nbytes+3*result["absorbed"].size)
    d["direction_cell_updates"] *= 2
    d["dense_equivalent_direction_cell_updates"] = d["direction_cell_updates"]
    d["direction_cell_updates"] = int(backend.get_active_updates())
    d["optimization"] = ("Exact zero-radiation activity masks; no intensity threshold or pruning"
                         if active_cells else "Activity masking disabled for dense reference")
    return result
