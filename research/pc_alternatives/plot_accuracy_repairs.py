"""Plot measured component convergence; each axis states its own error metric."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE/name).read_text())


def main():
    angular=read('validation_angular_accuracy.json')
    coupled=read('validation_coupled_opacity.json')
    history=read('validation_coupled_3d.json')
    finite=read('validation_finite_c.json')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                         'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig, axes=plt.subplots(2,2,figsize=(11,7.8),layout='constrained')
    blue,orange,green='#245C9F','#C75B12','#16806A'
    rows=angular['spatial_sequence']['rows']; ax=axes[0,0]
    x=np.array([r['radius_cells'] for r in rows]); y=np.array([r['continuum_atom_l1_over_Q'] for r in rows])*100
    ax.loglog(x,y,'o-',color=blue,label='Angular-controlled sharp front')
    ax.scatter([8],[100*angular['no_escape_sphere']['low_K48_continuum_atom_l1_over_Q']],color=orange,marker='x',s=75,label='Previous K=48 (no escape)')
    ax.set(xlabel='Sphere radius / cell width',ylabel='Atom L1 / emitted photons (%)',title='A. Angle repair exposes spatial truncation')
    ax.set_xticks(x,labels=[str(int(v)) for v in x]); ax.legend(fontsize=8)
    rows=history['rows']; ax=axes[0,1]
    x=np.array([r['subdivisions_per_source_interval'] for r in rows]); y=np.array([r['max_x_error'] for r in rows])
    ax.loglog(x,y,'o-',color=green,label='Three oblique beams + chemistry')
    ax.loglog(x,y[0]/x**2,'--',color='0.55',label='Second-order reference')
    ax.set(xlabel='Subdivisions per source-history interval',ylabel='Maximum ionized-fraction error',title='B. Time history of simultaneous sources')
    ax.set_xticks(x,labels=[str(int(v)) for v in x]); ax.legend(fontsize=8)
    rows=finite['refinement']['spatial']['rows']; ax=axes[1,0]
    x=np.array([r['cell_width'] for r in rows]); y=np.abs([r['relative_radius_error'] for r in rows])*100
    ax.loglog(x,y,'o-',color=blue)
    ax.set(xlabel='Cell width (fixed physical sphere)',ylabel='Ionized-volume radius error (%)',title='C. Finite-c front: voxel-entry bias remains')
    ax.invert_xaxis()
    rows=coupled['spatial']['recombining_slab']['rows']; ax=axes[1,1]
    x=np.array([r['cells'] for r in rows]); y=np.array([r['l1_ionized_atoms_error'] for r in rows])
    ax.loglog(x,y,'o-',color=green,label='Recombining finite-opacity slab')
    ax.loglog(x,y[0]*(x[0]/x)**2,'--',color='0.55',label='Second-order reference')
    ax.set(xlabel='Cells across slab (fixed physical length)',ylabel='Ionized-atom L1 error',title='D. Spatial chemistry/attenuation convergence')
    ax.set_xticks(x,labels=[str(int(v)) for v in x]); ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2,which='major')
    fig.suptitle('Accuracy after targeted repairs',fontsize=17,fontweight='bold')
    folder=HERE/'figures'; folder.mkdir(exist_ok=True)
    for suffix in ('png','pdf'):
        fig.savefig(folder/f'accuracy_repairs_convergence.{suffix}')
    plt.close(fig)


if __name__=='__main__':
    main()
