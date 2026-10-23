#ifndef _RECOMB_H
#define _RECOMB_H

double splined_recombination_rate(double z_eff, double gamma12_bg);
/* Same table and limits, with no shared mutable interpolation accelerator. */
double splined_recombination_rate_no_accel(double z_eff, double gamma12_bg);
/*initializes the lookup table for the PDF density integral in MHR00 model at redshift z*/
void init_MHR();
void free_MHR();

#endif
