#!/usr/bin/env python3
"""Research-only adapter from real Qin20 SFRD fields to positive UV rate increments.

The native effective-count convention is explicit and kept separate from a
physical photon count per hydrogen atom. No cumulative nion map is used as a
rate, differenced locally, reset, or clipped. Exact endpoint fields can be
reconstructed from the frozen ICs with only the short prefix through z=8.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time

import h5py
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[2]


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def constants_from_source():
    text=(ROOT/"src/py21cmfast/src/Constants.c").read_text()
    return {name:float(re.search(r"\."+name+r"\s*=\s*([0-9.eE+-]+)",text)[1])
            for name in ("G","Msun","m_p","cm_per_Mpc","s_per_yr")}


def mean_baryon_density(cosmo, constants):
    """Match Constants.h: comoving baryon mass density in Msun/cMpc^3."""
    h0=cosmo["hlittle"]*3.2407e-18
    return 3*h0*h0/(8*np.pi*constants["G"])*constants["cm_per_Mpc"]**3/constants["Msun"]*cosmo["OMb"]


def uv_rates(sfrd_acg, sfrd_mcg, *, astro, rho_b):
    """Return escaped effective photon rates per mean-density cell, in s^-1.

    Native SFRD is Msun/s/cMpc^3 and ALREADY includes Eulerian rho. This
    Qin20 adapter requires mass-independent escape fractions (ALPHA_ESC=0).
    """
    a=np.asarray(sfrd_acg,dtype=float);m=np.asarray(sfrd_mcg,dtype=float)
    if a.shape!=m.shape or not np.isfinite(a).all() or not np.isfinite(m).all():
        raise ValueError("Matching finite SFRD arrays required")
    if np.any(a<0) or np.any(m<0) or not np.isfinite(rho_b) or rho_b<=0:
        raise ValueError("Nonnegative SFRD and positive mean baryon density required")
    if not all(np.isfinite(astro[key]) for key in ("ALPHA_ESC","F_ESC10_ACG","F_ESC7_MCG","POP2_ION","POP3_ION")):
        raise ValueError("Finite escape and yield parameters required")
    if astro["ALPHA_ESC"]!=0:
        raise ValueError("Mass-dependent fesc requires a weighted mass integral, not this adapter")
    fa=min(10.**astro["F_ESC10_ACG"],1.)
    fm=min(10.**astro["F_ESC7_MCG"],1.)
    if min(astro["POP2_ION"],astro["POP3_ION"])<0:
        raise ValueError("Nonnegative photon yields required")
    return dict(acg=a*fa*astro["POP2_ION"]/rho_b,
                mcg=m*fm*astro["POP3_ION"]/rho_b)


def positive_increment(left, right, dt_seconds):
    """Trapezoidal time integral for nonnegative endpoint rates; no clipping."""
    a=np.asarray(left,dtype=float);b=np.asarray(right,dtype=float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Matching finite rate arrays required")
    if np.any(a<0) or np.any(b<0) or not np.isfinite(dt_seconds) or dt_seconds<0:
        raise ValueError("Nonnegative rates and nonnegative finite duration required")
    return .5*dt_seconds*(a+b)


def time_interval(z_old,z_new,cosmo):
    """Native dtdz interval, with distinct native hubble/SFR clock reported.

    Qin20's OMl property is 1-OMm. Native dtdz omits radiation whereas
    hubble()/t_hubble(), used for SFR, includes it. These are two different
    native definitions, not quadrature uncertainty or an Astropy calculation.
    """
    if not z_old>z_new>=0:
        raise ValueError("Forward physical time requires decreasing nonnegative redshift")
    h0=cosmo["hlittle"]*3.2407e-18
    oml=cosmo.get("OMl",1-cosmo["OMm"])
    if cosmo["OMk"]!=0 or cosmo["wl"]!=-1:
        raise ValueError("This validation matches the flat LCDM native Qin20 clock only")
    def hubble_clock(z):
        return 1/((1+z)*h0*np.sqrt(cosmo["OMm"]*(1+z)**3+cosmo["OMr"]*(1+z)**4+oml))
    def clock(z):
        return np.sqrt(cosmo["OMm"]+oml)/((1+z)*h0*np.sqrt(cosmo["OMm"]*(1+z)**3+oml))
    exact,error=quad(clock,z_new,z_old,epsabs=1.,epsrel=1e-11)
    hubble_exact,hubble_error=quad(hubble_clock,z_new,z_old,epsabs=1.,epsrel=1e-11)
    return dict(integrated_seconds=exact,quadrature_error_seconds=error,
                definition="Native dtdz matter+Lambda clock; not native radiation-inclusive hubble()",
                native_endpoint_approximation_seconds=clock(z_new)*(z_old-z_new),
                native_hubble_integrated_seconds=hubble_exact,
                native_hubble_quadrature_error_seconds=hubble_error,
                dtdz_over_hubble_interval=exact/hubble_exact)


def helper_module():
    spec=importlib.util.spec_from_file_location("frozen_helpers",ROOT/"tools/validate_global_budget.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def extract_endpoints(fixture,cache):
    """Replay the exact saved short prefix; never replace prior feedback by current feedback."""
    from py21cmfast.drivers.coeval import generate_coeval
    from py21cmfast.drivers.single_field import compute_emissivity_fields
    from py21cmfast.io.caching import OutputCache
    from py21cmfast.io.h5 import write_output_to_hdf5

    helpers=helper_module()
    current,previous,frozen=helpers.load_frozen_step(fixture)
    inputs=current.inputs
    zprev=float(previous.ionized_box.redshift);znow=float(current.redshift)
    nodes=tuple(z for z in inputs.node_redshifts if z>=zprev)
    cache.mkdir(parents=True,exist_ok=True)
    initial=helpers.copy_initial_conditions(frozen,inputs)
    progress=[];started=time.perf_counter()
    for coeval,_ in generate_coeval(inputs=inputs,initial_conditions=initial,
            out_redshifts=nodes,regenerate=True,write=False,
            cache=OutputCache(cache/"native_cache"),progressbar=False):
        row=dict(redshift=float(coeval.redshift),elapsed_seconds=time.perf_counter()-started)
        progress.append(row);print(json.dumps(row),flush=True)
        if coeval.redshift==zprev:
            old_emissivity=coeval.emissivity_fields
            old_spin=coeval.ts_box
            differences={}
            for name in ("neutral_fraction","cumulative_recombinations","nion_conditional_filtered_acg","nion_conditional_filtered_mcg"):
                a=coeval.ionized_box.get(name);b=previous.ionized_box.get(name)
                differences[name]={"bitwise_equal":bool(np.array_equal(a,b)),
                                   "max_abs_difference":float(np.max(np.abs(a.astype(float)-b)))}
                assert np.allclose(a,b,rtol=1e-6,atol=2e-7),differences
            assert np.array_equal(coeval.perturbed_field.get("density"),previous.perturbed_field.get("density"))
            write_output_to_hdf5(old_emissivity,cache/"emissivity_previous.h5")
            write_output_to_hdf5(old_spin,cache/"spin_previous.h5")
            current_emissivity=compute_emissivity_fields(inputs=inputs,redshift=znow,
                initial_conditions=helpers.copy_initial_conditions(frozen,inputs),
                perturbed_field=helpers.copy_output(current.perturbed_field,inputs),
                previous_spin_temp=old_spin,
                previous_ionize_box=helpers.copy_output(previous.ionized_box,inputs),
                regenerate=True,write=False)
            write_output_to_hdf5(current_emissivity,cache/"emissivity_current.h5")
            metadata=dict(scope="Exact short-prefix replay from frozen ICs, then current instantaneous emissivity with correct previous feedback",
                redshift_previous=zprev,redshift_current=znow,nodes=list(nodes),
                progress=progress,replay_agreement=differences,
                density_bitwise_equal=True,total_seconds=time.perf_counter()-started)
            (cache/"extraction.json").write_text(json.dumps(metadata,indent=2)+"\n")
            return
    raise RuntimeError("Previous fixture redshift was not reached")


def read_emissivity(path):
    with h5py.File(path) as handle:
        group=handle["EmissivityFields"]
        return {key:group["OutputFields"][key][...].astype(float)
                for key in group["OutputFields"]},dict(group["OutputFields"].attrs)


def extract_held_feedback_quadrature(fixture,cache):
    """Sample one fixed z8-feedback family at t_left, midpoint and t_right."""
    from py21cmfast.drivers.single_field import compute_emissivity_fields, perturb_field
    from py21cmfast.io.h5 import read_output_struct, write_output_to_hdf5
    helpers=helper_module()
    current,previous,frozen=helpers.load_frozen_step(fixture)
    old_spin=read_output_struct(cache/"spin_previous.h5");old_spin.load_all()
    cosmo=current.inputs.asdict(include_cosmo_tables="never",only_cstruct_params=False)["cosmo_params"]
    zo=float(previous.ionized_box.redshift);zn=float(current.redshift)
    duration=time_interval(zo,zn,cosmo)["integrated_seconds"]
    zm=brentq(lambda z:time_interval(zo,z,cosmo)["integrated_seconds"]-duration/2,
              zn,zo-1e-10,xtol=1e-12)
    started=time.perf_counter()
    epsilon=1e-6
    for name,z in (("held_previous",zo-epsilon),("midpoint",zm)):
        density=perturb_field(redshift=z,inputs=current.inputs,
            initial_conditions=helpers.copy_initial_conditions(frozen,current.inputs),
            regenerate=True,write=False)
        if name=="midpoint":
            write_output_to_hdf5(density,cache/"density_midpoint.h5")
        box=compute_emissivity_fields(inputs=current.inputs,redshift=z,
            initial_conditions=helpers.copy_initial_conditions(frozen,current.inputs),
            perturbed_field=density,previous_spin_temp=old_spin,
            previous_ionize_box=helpers.copy_output(previous.ionized_box,current.inputs),
            regenerate=True,write=False)
        write_output_to_hdf5(box,cache/f"emissivity_{name}.h5")
    metadata=dict(redshift_midpoint=zm,feedback_redshift=zo,
        redshift_left_evaluation=zo-epsilon,left_limit_redshift_offset=epsilon,
        left_limit_time_offset_seconds=time_interval(zo,zo-epsilon,cosmo)["integrated_seconds"],
        policy="Hold z8 TsBox and IonizedBox for left/mid/right emissivity evaluations; left is the one-sided z8-minus-1e-6 value with its actual perturbed density. Native exact z8 chronology instead uses z12 feedback. This is a quadrature diagnostic under held feedback, not refined feedback evolution.",
        elapsed_seconds=time.perf_counter()-started)
    (cache/"held_feedback.json").write_text(json.dumps(metadata,indent=2)+"\n")


def run(args):
    args.cache.mkdir(parents=True,exist_ok=True)
    fixture_hashes={p.name:file_hash(p) for p in args.fixture.glob("*.h5")}
    if not (args.cache/"extraction.json").exists():
        if not args.extract:
            raise RuntimeError("Native emissivity fields absent; use --extract for the short-prefix replay")
        extract_endpoints(args.fixture,args.cache)
    if not (args.cache/"held_feedback.json").exists():
        if not args.extract:
            raise RuntimeError("Held-feedback quadrature fields absent; use --extract")
        extract_held_feedback_quadrature(args.fixture,args.cache)
    extraction=json.loads((args.cache/"extraction.json").read_text())
    old,_=read_emissivity(args.cache/"emissivity_previous.h5")
    new,_=read_emissivity(args.cache/"emissivity_current.h5")
    with h5py.File(args.fixture/"previous_ionized.h5") as h:
        parameters=h["IonizedBox/InputParameters"]
        astro=dict(parameters["astro_params"].attrs);cosmo=dict(parameters["cosmo_params"].attrs)
        sim=dict(parameters["simulation_options"].attrs)
        old_unconditional={name:float(h["IonizedBox/OutputFields"].attrs[f"nion_unconditional_{name}"]) for name in ("acg","mcg")}
    with h5py.File(args.fixture/"previous_density.h5") as h:
        rho_old=1+h["PerturbedField/OutputFields/density"][...].astype(float)
    with h5py.File(args.fixture/"density.h5") as h:
        rho_new=1+h["PerturbedField/OutputFields/density"][...].astype(float)
    native=json.loads((ROOT/"research/pc_alternatives/native_benchmark.json").read_text())
    previous_json=native["preparation"]
    assert previous_json["grid"]==sim["HII_DIM"] and previous_json["seed"]==1234
    latest=[x for x in native["records"] if x["mode"]=="no-photoncons"][0]
    new_unconditional={name:latest[f"source_{name}"] for name in ("acg","mcg")}
    constants=constants_from_source();rhob=mean_baryon_density(cosmo,constants)
    old_rates=uv_rates(old["sfrd_acg"],old["sfrd_mcg"],astro=astro,rho_b=rhob)
    new_rates=uv_rates(new["sfrd_acg"],new["sfrd_mcg"],astro=astro,rho_b=rhob)
    held,_=read_emissivity(args.cache/"emissivity_held_previous.h5")
    midpoint,_=read_emissivity(args.cache/"emissivity_midpoint.h5")
    held_rates=uv_rates(held["sfrd_acg"],held["sfrd_mcg"],astro=astro,rho_b=rhob)
    midpoint_rates=uv_rates(midpoint["sfrd_acg"],midpoint["sfrd_mcg"],astro=astro,rho_b=rhob)
    held_metadata=json.loads((args.cache/"held_feedback.json").read_text())
    clock=time_interval(extraction["redshift_previous"],extraction["redshift_current"],cosmo)
    ncell=rho_new.size;delta=np.zeros_like(rho_new);species={}
    for name in ("acg","mcg"):
        increment=positive_increment(old_rates[name],new_rates[name],clock["integrated_seconds"])
        delta+=increment
        legacy_delta=ncell*(new_unconditional[name]-old_unconditional[name])
        # Native SFRD mean-fix normalizes the conditional (SFRD/rho) mean.
        normalized_old=old_rates[name]*old_rates[name].size*np.mean(old_rates[name]/rho_old)/old_rates[name].sum()
        normalized_new=new_rates[name]*new_rates[name].size*np.mean(new_rates[name]/rho_new)/new_rates[name].sum()
        global_normalized_increment=positive_increment(normalized_old,normalized_new,clock["integrated_seconds"])
        species[name]=dict(rate_old_total=float(old_rates[name].sum()),rate_new_total=float(new_rates[name].sum()),
            rate_old_min=float(old_rates[name].min()),rate_new_min=float(new_rates[name].min()),
            sum_rate_over_density_mean_budget_old=float(old_rates[name].sum()/(ncell*np.mean(old_rates[name]/rho_old))),
            sum_rate_over_density_mean_budget_new=float(new_rates[name].sum()/(ncell*np.mean(new_rates[name]/rho_new))),
            integrated_effective_photons=float(increment.sum()),
            original_left_rate_held_for_interval=float(old_rates[name].sum()*clock["integrated_seconds"]),
            original_right_rate_held_for_interval=float(new_rates[name].sum()*clock["integrated_seconds"]),
            minimum_cell_increment=float(increment.min()),negative_cells=int(np.count_nonzero(increment<0)),
            legacy_cumulative_photon_increment=legacy_delta,
            rate_integral_over_legacy_increment=float(increment.sum()/legacy_delta),
            independently_global_normalized_rate_integral=float(global_normalized_increment.sum()),
            global_normalized_rate_integral_over_legacy_increment=float(global_normalized_increment.sum()/legacy_delta))
        fixed_trapezoid=positive_increment(held_rates[name],new_rates[name],clock["integrated_seconds"])
        fixed_simpson=clock["integrated_seconds"]*(held_rates[name]+4*midpoint_rates[name]+new_rates[name])/6
        species[name]["held_feedback_quadrature"]=dict(
            left_rate_total=float(held_rates[name].sum()),midpoint_rate_total=float(midpoint_rates[name].sum()),
            right_rate_total=float(new_rates[name].sum()),trapezoid_integral=float(fixed_trapezoid.sum()),
            simpson_integral=float(fixed_simpson.sum()),
            simpson_over_trapezoid=float(fixed_simpson.sum()/fixed_trapezoid.sum()),
            simpson_over_legacy_increment=float(fixed_simpson.sum()/legacy_delta),
            left_feedback_change_fraction=float(held_rates[name].sum()/old_rates[name].sum()-1))
    identity_checks={}
    for name,z,rates,emissivity,rho in (("previous",extraction["redshift_previous"],old_rates,old,rho_old),
                                      ("current",extraction["redshift_current"],new_rates,new,rho_new)):
        h=cosmo["hlittle"]*3.2407e-18*np.sqrt(cosmo["OMm"]*(1+z)**3+cosmo["OMr"]*(1+z)**4+cosmo.get("OMl",1-cosmo["OMm"]))
        timescale=astro["t_STAR"]/h
        predicted=(rates["acg"]+rates["mcg"])/rho*timescale
        expected=emissivity["n_ion"]
        ratio=float(predicted.mean()/expected.mean())
        assert abs(ratio-1)<2e-6,ratio
        identity_checks[name]=dict(sfr_timescale_seconds=timescale,
            mean_rate_over_rho_times_timescale=float(predicted.mean()),
            instantaneous_emissivity_nion_mean=float(expected.mean()),ratio=ratio,
            cellwise_rms_error_over_native_nion_mean=float(np.sqrt(np.mean((predicted-expected)**2))/expected.mean()))
    cumulative_differences={}
    with h5py.File(args.fixture/"previous_ionized.h5") as h,np.load(args.fixture.parent/"source_capacity.npz") as current_packets:
        for name in ("acg","mcg"):
            previous_raw=rho_old*h[f"IonizedBox/OutputFields/nion_conditional_filtered_{name}"][0].astype(float)
            previous_packets=previous_raw*(ncell*old_unconditional[name]/previous_raw.sum())
            difference=current_packets[f"source_{name}"]-previous_packets
            cumulative_differences[name]=dict(minimum_increment=float(difference.min()),
                negative_cells=int(np.count_nonzero(difference<0)),negative_total=float(-difference[difference<0].sum()),
                global_increment=float(difference.sum()),
                interpretation="Difference of separately budget-normalized Eulerian cumulative packet closures; not an emission rate")
    # Independent physical-count route checks mass, volume and time conversion.
    volume=(sim["BOX_LEN"]/sim["HII_DIM"])**3
    mean_nucleons_per_cell=rhob*volume*constants["Msun"]/constants["m_p"]
    photons_from_rates=delta*mean_nucleons_per_cell
    weighted_sfr_old=old["sfrd_acg"]*min(10.**astro["F_ESC10_ACG"],1.)*astro["POP2_ION"]+old["sfrd_mcg"]*min(10.**astro["F_ESC7_MCG"],1.)*astro["POP3_ION"]
    weighted_sfr_new=new["sfrd_acg"]*min(10.**astro["F_ESC10_ACG"],1.)*astro["POP2_ION"]+new["sfrd_mcg"]*min(10.**astro["F_ESC7_MCG"],1.)*astro["POP3_ION"]
    photons_direct=positive_increment(weighted_sfr_old,weighted_sfr_new,clock["integrated_seconds"])*volume*constants["Msun"]/constants["m_p"]
    conversion_error=float(np.max(np.abs(photons_from_rates-photons_direct))/np.max(photons_direct))
    assert conversion_error<2e-15
    legacy_total=sum(x["legacy_cumulative_photon_increment"] for x in species.values())
    held_total_trapezoid=sum(x["held_feedback_quadrature"]["trapezoid_integral"] for x in species.values())
    held_total_simpson=sum(x["held_feedback_quadrature"]["simpson_integral"] for x in species.values())
    result=dict(scope="Validated Qin20 instantaneous SFRD adapter and one coarse chronological interval; no transport/chemistry coupling",
        fixture=str(args.fixture),cache=str(args.cache),fixture_sha256=fixture_hashes,
        auxiliary_inputs_sha256={str(path):file_hash(path) for path in
            (args.fixture.parent/"source_capacity.npz",ROOT/"research/pc_alternatives/native_benchmark.json")},
        extraction=extraction,available_native_fields=sorted(new),
        missing_saved_inputs_initially=["EmissivityFields(z8)","EmissivityFields(z7.5)","TsBox(z8)"],
        native_source_files_sha256={name:file_hash(ROOT/"src/py21cmfast/src"/name) for name in ("Constants.c","Constants.h","EmissivityFields.c","hmf.c","scaling_relations.c","IonisationBox.c")},
        template_physical_comparison=previous_json["physical_differences_from_template"],
        adapter_parameters={name:float(astro[name]) for name in ("t_STAR","F_ESC10_ACG","F_ESC7_MCG","ALPHA_ESC","POP2_ION","POP3_ION")},
        units=dict(sfrd="Msun / second / comoving Mpc^3; Eulerian density already included",
            effective_rate="1/second in source normalized to mean baryonic mass per cell",
            effective_count="emitted photons divided by mean nucleons per cell under POP yield per stellar nucleon",
            rho_b_Msun_per_cMpc3=rhob,physical_constants=constants,
            emitted_photons_per_mean_hydrogen_atom_conversion_factor=1/(1-cosmo["Y_He"]),
            mean_nucleons_per_cell=mean_nucleons_per_cell,
            independent_mass_volume_conversion_relative_error=conversion_error,
            physical_interpretation_caveat="Per-nucleon POP interpretation follows stellar mass normalization; native N_b0 denotes H+He nuclei, while No is H only. Do not silently equate effective native ledger to physical H absorption."),
        clock={**clock,"integrated_Myr":clock["integrated_seconds"]/constants["s_per_yr"]/1e6,
            "native_hubble_integrated_Myr":clock["native_hubble_integrated_seconds"]/constants["s_per_yr"]/1e6},
        native_instantaneous_rate_identity=identity_checks,
        cumulative_packet_difference_diagnostic=cumulative_differences,
        held_feedback_quadrature_policy=held_metadata,
        held_feedback_total=dict(trapezoid_integral=held_total_trapezoid,simpson_integral=held_total_simpson,
            simpson_over_trapezoid=held_total_simpson/held_total_trapezoid,
            simpson_over_legacy_increment=held_total_simpson/legacy_total),
        endpoint_integral_caveat="Left-held and right-held numbers are diagnostics, not rigorous bounds without monotonicity throughout the interval; original endpoint rates use different prior-feedback histories",
        species=species,total_rate_integral=float(delta.sum()),legacy_cumulative_increment=legacy_total,
        total_rate_integral_over_legacy_increment=float(delta.sum()/legacy_total),
        total_rate_integral_with_native_hubble_interval=float(delta.sum()/clock["dtdz_over_hubble_interval"]),
        native_endpoint_dt_rate_integral=float(delta.sum()*clock["native_endpoint_approximation_seconds"]/clock["integrated_seconds"]),
        no_clipping=True,template_parameters_modified=False,
        temporal_photon_model_changed_relative_to_legacy=True,
        temporal_closure_matches_legacy=False,
        limitations=["Trapezoidal rate integral over one coarse interval is not a time-converged photon history",
            "Native instantaneous emissivity uses previous spin feedback; ionization source uses current spin feedback",
            "Direct SFRD volume sum differs from native conditional-mean normalization",
            "Actual absorbed H photons require explicit helium and secondary-ionization treatment",
            "No recombination, transport, density advection, pending-ray continuation or photoionization-rate coupling validated here"])
    after={p.name:file_hash(p) for p in args.fixture.glob("*.h5")}
    assert after==fixture_hashes
    result["frozen_inputs_unchanged"]=True
    np.savez_compressed(args.cache/"validated_rate_adapter.npz",rate_old=old_rates["acg"]+old_rates["mcg"],rate_new=new_rates["acg"]+new_rates["mcg"],emission_increment=delta)
    args.output.write_text(json.dumps(result,indent=2,default=lambda x:x.item() if hasattr(x,"item") else str(x))+"\n")
    print(json.dumps({"species":species,"clock":result["clock"],"total_ratio":result["total_rate_integral_over_legacy_increment"]},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture",type=Path,default=Path("/workspace/scratch/40b418003e4a/qin20_native_cost_grid32/frozen_native_step"))
    parser.add_argument("--cache",type=Path,default=Path("/workspace/scratch/40b418003e4a/validation_native_rates_grid32"))
    parser.add_argument("--output",type=Path,default=Path(__file__).with_suffix(".json"))
    parser.add_argument("--extract",action="store_true")
    run(parser.parse_args())
