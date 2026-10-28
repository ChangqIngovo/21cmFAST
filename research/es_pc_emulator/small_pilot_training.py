"""CPU-sized history/chemistry/context/P21 ablations on existing pilot fields.

No native simulations run here. All splits are whole histories. RT thermal and
helium fields are auxiliary labels, never deployment inputs. Checkpoints retain
train-only normalization and expose ``predict_checkpoint`` for field inference.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from research.es_pc_emulator.history_objective import (
    SELECTION_METRIC,
    STAGE_EDGES,
    denominator_floor,
    relative_errors,
    stage_weights,
    weighted_rows,
)
from research.es_pc_emulator.physical_pilot_training import load_case, load_manifest
from research.es_pc_emulator.small_pilot_model import (
    FieldNet,
    build_inputs,
    power_loss,
    shell_power,
    thermal_targets,
)
from research.es_pc_emulator.spectral_components import component_loss, decompose_power
from research.pc_field_emulator.evaluation import field_metrics, shell_spectra

DEFAULT_ARMS = ("local", "history", "thermo", "spatial", "p21", "direct_p21")
COMPONENT_GROUPS = {
    "thermo_pxx": (1,),
    "thermo_cross": (2,),
    "thermo_nonlinear": (3, 4, 5),
    "thermo_components": tuple(range(6)),
    "thermo_components_p21": tuple(range(6)),
}
ARMS = (*DEFAULT_ARMS, "thermo_p21", *COMPONENT_GROUPS)
COMPONENT_NAMES = ("density_auto", "neutral_auto", "density_neutral_cross",
                   "product_auto", "density_product_cross", "neutral_product_cross")


def write_json(path, value):
    """Write a finite, human-readable diagnostic record."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def distribution(values):
    """Summarize finite errors while counting undefined measurements."""
    values = list(values)
    finite = np.array([v for v in values if v is not None and np.isfinite(v)])
    return {"count": len(values), "undefined": len(values) - len(finite),
            **{name: float(fn(finite)) if len(finite) else None for name, fn in
               (("mean", np.mean), ("median", np.median),
                ("p95", lambda x: np.quantile(x, .95)), ("max", np.max))}}


def select_cases(manifest, limits):
    """Select whole cases and balance external initial-condition seeds."""
    selected = []
    for split, count in limits.items():
        records = sorted((r for r in manifest["cases"] if r["split"] == split),
                         key=lambda r: r["case_id"])
        if split != "external_test" and not records:
            raise ValueError(f"Missing {split} cases")
        if len(records) < count:
            raise ValueError(f"Requested {count} {split} cases, but manifest has {len(records)}")
        if split == "external_test":
            # Round-robin IC seeds rather than taking only the first seed's cases.
            groups = {seed: [r for r in records if r["seed"] == seed]
                      for seed in sorted({r["seed"] for r in records})}
            records = [group[index] for index in range(max(map(len, groups.values()), default=0))
                       for group in groups.values() if index < len(group)]
        selected.extend(records[:count])
    cases = [load_case(record) for record in selected]
    reference = cases[0]
    for case in cases:
        if (case["xhi_es"].shape != reference["xhi_es"].shape
                or not np.array_equal(case["redshifts"], reference["redshifts"])
                or case["metadata"]["box_len_cMpc"] != reference["metadata"]["box_len_cMpc"]
                or case["metadata"]["parameter_names"] != reference["metadata"]["parameter_names"]):
            raise ValueError("Small pilot requires common grids, redshifts, box and parameter definitions")
        if len(case["redshifts"]) < 3:
            raise ValueError("Primary evaluation requires at least three snapshots")
        if not np.isfinite(case["metadata"]["theta"]).all():
            raise ValueError("Nonfinite source-control parameters")
    return cases


def density_at(case, snapshot):
    """Return the nonnegative density field at one snapshot."""
    rho = np.asarray(case["density"], dtype=np.float32)
    rho = rho[snapshot] if rho.ndim == 4 else rho
    if rho.shape != case["xhi_es"].shape[1:] or not np.isfinite(rho).all() or np.any(rho < 0):
        raise ValueError("Invalid density")
    return rho


def snapshot_mask(count, snapshot_scope="last-three"):
    """Select the declared spectral/validation snapshots without changing field fitting."""
    if snapshot_scope not in ("last-three", "all"):
        raise ValueError("snapshot_scope must be last-three or all")
    return np.ones(count, dtype=bool) if snapshot_scope == "all" else np.arange(count) >= count-3


def field_arrays(cases, history_steps, auxiliary, snapshot_scope="last-three"):
    """Build full-cube batches with separate deployment inputs and labels."""
    rows = {name: [] for name in ("inputs", "base", "truth", "brightness", "density",
                                 "amplitude", "aux", "primary", "teacher_mean")}
    names = None
    for case in cases:
        selected_snapshots = snapshot_mask(len(case["redshifts"]), snapshot_scope)
        for snapshot, redshift in enumerate(case["redshifts"]):
            inputs, columns = build_inputs(case, snapshot, history_steps=history_steps)
            if names is not None and names != columns:
                raise ValueError("Feature columns changed between cases")
            names = columns
            rows["inputs"].append(inputs)
            rows["base"].append(case["xhi_es"][snapshot])
            rows["truth"].append(case["xhi_rt"][snapshot])
            rows["teacher_mean"].append(np.mean(case["xhi_rt"][snapshot], dtype=np.float64))
            rows["primary"].append(selected_snapshots[snapshot])
            amplitude = 27 * np.sqrt((1 + redshift) / 10)
            density = density_at(case, snapshot)
            rows["brightness"].append(amplitude * density)
            rows["density"].append(density)
            rows["amplitude"].append(amplitude)
            if auxiliary:
                rows["aux"].append(thermal_targets(case, snapshot))
    arrays = {key: torch.from_numpy(np.asarray(value, dtype=np.float64 if key == "teacher_mean" else np.float32))
              for key, value in rows.items() if value}
    if not all(torch.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Nonfinite training arrays")
    return arrays, names


def normalization(inputs):
    """Compute channel normalization from the supplied training cubes."""
    axes = (0, 2, 3, 4)
    mean = inputs.mean(dim=axes, keepdim=True)
    scale = inputs.std(dim=axes, keepdim=True, unbiased=False).clamp_min(1e-3)
    return mean, scale


def primary_band(k, dimension, box):
    """Select the common physical modes used for fitting and validation."""
    mask = (k >= .1) & (k <= .6 * np.pi * dimension / box)
    if not mask.any():
        raise ValueError("No primary spectral bins")
    return mask


def training_component_scale(train, box, floor):
    """Compute fixed per-term/per-mode RMS scales from primary training labels."""
    mask = train["primary"].bool()
    if not mask.any():
        raise ValueError("Component scaling requires primary training snapshots")
    with torch.no_grad():
        terms = decompose_power(train["density"][mask], train["truth"][mask],
                                train["amplitude"][mask], box)["terms"]
        weights = train.get("sample_weight", torch.ones(len(train["truth"])))[mask]
        scale = (terms.square()*weights[:, None, None]).mean(dim=0).sqrt().clamp_min(floor)
    if not torch.isfinite(scale).all() or not torch.all(scale > 0):
        raise ValueError("Invalid training component scales")
    return scale


def field_loss(model, batch, power_weight, auxiliary_weight, power_floor, box, *,
               primary_spectra=False, component_indices=(), component_scale=None,
               component_weight=0., power_denominator: torch.Tensor | None = None):
    """Combine field, chemistry and spectral errors.

    ``power_denominator`` enables the stage-weighted relative-power objective;
    its positive per-mode scales must come from training labels only.
    """
    prediction, auxiliary = model(batch["inputs"], batch["base"])
    weights = batch.get("sample_weight", torch.ones(len(prediction), device=prediction.device))
    loss = weighted_rows((prediction - batch["truth"]) ** 2, weights)
    if auxiliary is not None:
        # Temperature is log10(K); helium fractions have a simplex constraint.
        loss = loss + auxiliary_weight * weighted_rows((auxiliary - batch["aux"]) ** 2, weights)
    spectral_rows = batch["primary"].bool() if primary_spectra else torch.ones(
        len(prediction), dtype=torch.bool, device=prediction.device)
    if power_weight and spectral_rows.any():
        k, predicted, _ = shell_power(batch["brightness"][spectral_rows] * prediction[spectral_rows], box)
        _, truth, _ = shell_power(batch["brightness"][spectral_rows] * batch["truth"][spectral_rows], box)
        if primary_spectra:
            band = primary_band(k, prediction.shape[-1], box)
            predicted, truth = predicted[..., band], truth[..., band]
        if power_denominator is None:
            spectral_loss = power_loss(predicted, truth, floor=power_floor)
        else:
            denominator = power_denominator[band] if primary_spectra else power_denominator
            spectral_loss = weighted_rows(relative_errors(predicted, truth, denominator),
                                          weights[spectral_rows])
        loss = loss + power_weight * spectral_loss
    if component_indices and spectral_rows.any():
        predicted = decompose_power(batch["density"][spectral_rows], prediction[spectral_rows],
                                    batch["amplitude"][spectral_rows], box)
        truth = decompose_power(batch["density"][spectral_rows], batch["truth"][spectral_rows],
                                batch["amplitude"][spectral_rows], box)
        band = primary_band(predicted["k"], prediction.shape[-1], box)
        if "sample_weight" in batch:
            scaled = (torch.asinh(predicted["terms"][..., band]/component_scale[..., band])
                      - torch.asinh(truth["terms"][..., band]/component_scale[..., band])).square()
            penalty = weighted_rows(scaled[:, component_indices, :], weights[spectral_rows])
        else:
            penalty = component_loss(predicted["terms"][..., band], truth["terms"][..., band],
                                     component_scale[..., band], component_indices)
        loss = loss + component_weight * penalty
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training/validation loss")
    return loss


@torch.no_grad()
def field_diagnostics(model, arrays, box, power_floor, component_scale, batch_size):
    """Measure comparable train/validation metrics on primary snapshots and bins.

    All reported field and chemistry errors use those same snapshots. Component
    loss includes all six terms with fixed training-only scales, irrespective of
    which terms enter optimization. Relative power errors exclude undefined bins.
    """
    if batch_size < 1:
        raise ValueError("Diagnostic batch size must be positive")
    indices = torch.nonzero(arrays["primary"].bool(), as_tuple=False).flatten()
    if not len(indices):
        raise ValueError("Diagnostics require primary snapshots")
    sums = dict.fromkeys(("log_power_mse", "field_mae", "field_mse", "component_loss"), 0.)
    relative_sum, valid_count, bins, auxiliary_sum = 0., 0, 0, 0.
    has_auxiliary = False
    for selected in indices.split(batch_size):
        batch = {key: value[selected] for key, value in arrays.items()}
        prediction, auxiliary = model(batch["inputs"], batch["base"])
        prediction, truth = prediction.double(), batch["truth"].double()
        density, amplitude = batch["density"].double(), batch["amplitude"].double()
        k, pp, _ = shell_power(batch["brightness"].double() * prediction, box)
        _, pt, valid = shell_power(batch["brightness"].double() * truth, box)
        band = primary_band(k, prediction.shape[-1], box)
        pp, pt, valid = pp[..., band], pt[..., band], valid[..., band]
        differences = prediction-truth
        predicted_terms = decompose_power(density, prediction, amplitude, box)["terms"][..., band]
        true_terms = decompose_power(density, truth, amplitude, box)["terms"][..., band]
        values = {"log_power_mse": power_loss(pp, pt, floor=power_floor),
                  "field_mae": differences.abs().mean(), "field_mse": differences.square().mean(),
                  "component_loss": component_loss(predicted_terms, true_terms, component_scale[..., band])}
        if any(not torch.isfinite(value) for value in values.values()):
            raise FloatingPointError("Nonfinite diagnostic metric")
        for key, value in values.items():
            sums[key] += float(value) * len(selected)
        relative_sum += float((pp[valid]/pt[valid]-1).abs().sum())
        valid_count += int(valid.sum())
        bins += valid.numel()
        if auxiliary is not None:
            has_auxiliary = True
            auxiliary_sum += float((auxiliary.double()-batch["aux"].double()).square().mean()) * len(selected)
    result = {key: value/len(indices) for key, value in sums.items()}
    result.update(power_fractional_error_mean=relative_sum/valid_count if valid_count else None,
                  power_fractional_error_undefined=bins-valid_count, power_bins=bins,
                  primary_snapshots=len(indices),
                  auxiliary_mse=auxiliary_sum/len(indices) if has_auxiliary else None)
    if any(value is not None and not np.isfinite(value) for value in result.values()):
        raise FloatingPointError("Nonfinite diagnostic metric")
    return result


@torch.no_grad()
def stage_relative_score(model: nn.Module, arrays: dict[str, torch.Tensor], box: float,
                         floor: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Score all snapshots with the exact stage-relative spectral objective.

    Parameters
    ----------
    model : torch.nn.Module
        Field network to score without gradients.
    arrays : dict of torch.Tensor
        Full split arrays, including globally normalized ``sample_weight``.
    box : float
        Comoving box length in Mpc.
    floor : torch.Tensor
        Fixed training-only denominator floors for every shell.
    batch_size : int
        Maximum snapshots per forward pass.

    Returns
    -------
    torch.Tensor
        Stage-weighted mean absolute regularized fractional power error.
    """
    if batch_size < 1 or not arrays["primary"].bool().all():
        raise ValueError("Stage scoring requires all snapshots and a positive batch size")
    total = torch.zeros((), dtype=torch.float64)
    count = len(arrays["truth"])
    for start in range(0, count, batch_size):
        batch = {key: value[start:start+batch_size] for key, value in arrays.items()}
        prediction, _ = model(batch["inputs"], batch["base"])
        k, predicted, _ = shell_power(batch["brightness"].double()*prediction.double(), box)
        _, truth, _ = shell_power(batch["brightness"].double()*batch["truth"].double(), box)
        band = primary_band(k, prediction.shape[-1], box)
        score = weighted_rows(relative_errors(predicted[..., band], truth[..., band], floor[band]),
                              batch["sample_weight"])
        total += score*len(prediction)
    return total/count


def train_loop(model, train, validation, loss_function, args, seed, validation_function=None,
               diagnostic_function=None):
    """Optimize batches and retain the best validation-selected parameters."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    best, best_state, stale = float("inf"), None, 0
    history = []
    diagnostic_scope = (f"{getattr(args, 'snapshot_scope', 'last-three')} snapshots and primary k bins; "
                        "raw diagnostics retain equal snapshot weighting")
    for epoch in range(args.epochs):
        model.train()
        indices = torch.randperm(len(next(iter(train.values()))), generator=generator)
        training_loss, count = 0., 0
        for batch_indices in indices.split(args.batch_size):
            batch = {key: value[batch_indices] for key, value in train.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            training_loss += float(loss.detach()) * len(batch_indices)
            count += len(batch_indices)
        model.eval()
        with torch.no_grad():
            if validation_function is not None:
                val_loss = float(validation_function(model, validation))
            else:
                val_loss = 0.
                n_validation = len(next(iter(validation.values())))
                for start in range(0, n_validation, args.batch_size):
                    batch = {key: value[start:start + args.batch_size] for key, value in validation.items()}
                    val_loss += float(loss_function(model, batch)) * len(next(iter(batch.values())))
                val_loss /= n_validation
            if not np.isfinite(val_loss):
                raise FloatingPointError("Nonfinite validation score")
        history.append({"epoch": epoch + 1, "train_loss": training_loss / count, "validation_loss": val_loss})
        if getattr(args, "training_objective", "legacy") == "stage-relative":
            history[-1]["selection_score"] = val_loss
        if val_loss < best:
            best, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
            best_epoch = epoch + 1
        else:
            stale += 1
        interval = getattr(args, "diagnostic_interval", 0)
        if diagnostic_function is not None and interval and (
                epoch == 0 or (epoch+1) % interval == 0 or epoch+1 == args.epochs or stale >= args.patience):
            history[-1]["diagnostics"] = {
                "scope": diagnostic_scope,
                "train": diagnostic_function(model, train),
                "validation": diagnostic_function(model, validation)}
        if epoch == 0 or (epoch + 1) % 10 == 0 or stale >= args.patience:
            print(json.dumps(history[-1]), flush=True)  # noqa: T201
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    result = {"history": history, "best_epoch": best_epoch, "best_validation_loss": best}
    if diagnostic_function is not None and getattr(args, "diagnostic_interval", 0):
        result["selected_diagnostics"] = {
            "scope": "validation-selected checkpoint; " + diagnostic_scope,
            "train": diagnostic_function(model, train),
            "validation": diagnostic_function(model, validation)}
    return result


def direct_model(n_parameters, n_outputs, width):
    """Construct the small physical-control to log-power network."""
    return nn.Sequential(nn.Linear(n_parameters, width), nn.SiLU(),
                         nn.Linear(width, width), nn.SiLU(), nn.Linear(width, n_outputs))


def true_power(case):
    """Measure teacher power spectra using the shared NumPy estimator."""
    spectra = []
    for j, z in enumerate(case["redshifts"]):
        measured = shell_spectra({"teacher": 27 * np.sqrt((1 + z) / 10) * density_at(case, j)
                                 * case["xhi_rt"][j]}, case["metadata"]["box_len_cMpc"])
        spectra.append(measured["P_teacher"])
    return np.asarray(spectra, dtype=np.float32)


def fit_direct(cases, args, seed):
    """Train the direct spectrum model with train-only scaling."""
    training = [c for c in cases if c["metadata"]["split"] == "train"]
    validation = [c for c in cases if c["metadata"]["split"] == "validation"]
    x = torch.tensor(np.array([c["metadata"]["theta"] for c in training]), dtype=torch.float32)
    powers = np.array([true_power(c) for c in training])
    shape = powers.shape[1:]
    y = torch.tensor(np.log(powers.reshape(len(powers), -1) + args.power_floor))
    xmean, xscale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-3)
    ymean, yscale = y.mean(0), y.std(0, unbiased=False).clamp_min(.1)
    vx = torch.tensor(np.array([c["metadata"]["theta"] for c in validation]), dtype=torch.float32)
    vy = torch.tensor(np.log(np.array([true_power(c) for c in validation]).reshape(len(validation), -1)
                             + args.power_floor))
    model = direct_model(x.shape[1], y.shape[1], max(32, args.width * 4))
    train = {"x": (x - xmean) / xscale, "y": (y - ymean) / yscale}
    val = {"x": (vx - xmean) / xscale, "y": (vy - ymean) / yscale}
    result = train_loop(model, train, val, lambda m, b: torch.mean((m(b["x"]) - b["y"]) ** 2), args, seed)
    checkpoint = {"kind": "direct_p21", "state_dict": model.state_dict(), "width": max(32, args.width * 4),
                  "n_parameters": x.shape[1], "n_outputs": y.shape[1], "output_shape": list(shape),
                  "input_mean": xmean, "input_scale": xscale, "output_mean": ymean, "output_scale": yscale,
                  "parameter_names": training[0]["metadata"]["parameter_names"], "power_floor": args.power_floor,
                  "redshifts": training[0]["redshifts"].tolist(), "box_len_cMpc": training[0]["metadata"]["box_len_cMpc"]}
    return checkpoint, result


def fit_field(cases, arm, args, seed):
    """Train one field ablation with common validation power scoring."""
    history_steps = 0 if arm == "local" else args.history_steps
    auxiliary = arm in ("thermo", "spatial", "p21") or arm.startswith("thermo_")
    wide = arm in ("spatial", "p21")
    snapshot_scope = getattr(args, "snapshot_scope", "last-three")
    stage_relative = getattr(args, "training_objective", "legacy") == "stage-relative"
    if stage_relative and snapshot_scope != "all":
        raise ValueError("stage-relative requires snapshot-scope all")
    train, names = field_arrays([c for c in cases if c["metadata"]["split"] == "train"],
                                history_steps, auxiliary, snapshot_scope)
    validation, validation_names = field_arrays(
        [c for c in cases if c["metadata"]["split"] == "validation"],
        history_steps, auxiliary, snapshot_scope)
    if names != validation_names:
        raise ValueError("Training and validation features differ")
    mean, scale = normalization(train["inputs"])
    train["inputs"] = (train["inputs"] - mean) / scale
    validation["inputs"] = (validation["inputs"] - mean) / scale
    model = FieldNet(len(names), width=args.width, wide=wide, auxiliary=auxiliary,
                     output_transform=getattr(args, "output_transform", "legacy"),
                     prior_epsilon=getattr(args, "prior_epsilon", 1e-4))
    box = cases[0]["metadata"]["box_len_cMpc"]
    weight = args.power_weight if arm in ("p21", "thermo_p21", "thermo_components_p21") else 0.
    indices = COMPONENT_GROUPS.get(arm, ())
    stage_counts = {}
    power_denominator = None
    if stage_relative:
        for split, arrays in (("train", train), ("validation", validation)):
            arrays["sample_weight"], stage_counts[split] = stage_weights(
                arrays["teacher_mean"])
        with torch.no_grad():
            _, training_power, _ = shell_power(train["brightness"]*train["truth"], box)
            power_denominator = denominator_floor(training_power, train["sample_weight"],
                getattr(args, "relative_power_floor", 1e-4), args.power_floor)
    component_scale = training_component_scale(train, box, args.power_floor) if indices else None
    primary_spectra = arm.startswith("thermo_") or snapshot_scope == "all"
    diagnostic_function = None
    if getattr(args, "diagnostic_interval", 0):
        diagnostic_scale = (component_scale if component_scale is not None
                            else training_component_scale(train, box, args.power_floor))

        def diagnostic_function(network, batch):
            metrics = field_diagnostics(network, batch, box, args.power_floor, diagnostic_scale, args.batch_size)
            if stage_relative:
                metrics["selection_score"] = float(stage_relative_score(network, batch, box,
                                                                        power_denominator, args.batch_size))
            return metrics

    def validation_score(network, batch):
        # All field arms select epochs on the SAME observable and physical bins.
        if stage_relative:
            return stage_relative_score(network, batch, box, power_denominator, args.batch_size)
        selected = {key: value[batch["primary"].bool()] for key, value in batch.items()}
        predicted, _ = network(selected["inputs"], selected["base"])
        k, pp, _ = shell_power(selected["brightness"] * predicted, box)
        _, pt, _ = shell_power(selected["brightness"] * selected["truth"], box)
        mask = primary_band(k, predicted.shape[-1], box)
        return power_loss(pp[..., mask], pt[..., mask], floor=args.power_floor)

    result = train_loop(model, train, validation,
                        lambda m, b: field_loss(m, b, weight, args.auxiliary_weight, args.power_floor, box,
                                               primary_spectra=primary_spectra, component_indices=indices,
                                               component_scale=component_scale,
                                               component_weight=args.component_weight, power_denominator=power_denominator),
                        args, seed, validation_function=validation_score, diagnostic_function=diagnostic_function)
    result.update(selection_metric=SELECTION_METRIC if stage_relative else "validation log-P21 MSE",
                  stage_counts=stage_counts, stage_edges=list(STAGE_EDGES))
    checkpoint = {"kind": "field", "state_dict": model.state_dict(), "input_mean": mean, "input_scale": scale,
                  "feature_names": names, "history_steps": history_steps, "width": args.width,
                  "wide": wide, "auxiliary": auxiliary, "box_len_cMpc": box,
                  "grid_dimension": cases[0]["xhi_es"].shape[-1],
                  "power_weight": weight, "auxiliary_weight": args.auxiliary_weight, "power_floor": args.power_floor,
                  "primary_spectra": primary_spectra, "snapshot_scope": snapshot_scope,
                  "training_objective": getattr(args, "training_objective", "legacy"),
                  "output_transform": getattr(args, "output_transform", "legacy"),
                  "prior_epsilon": getattr(args, "prior_epsilon", 1e-4),
                  "power_denominator_floor": power_denominator,
                  "relative_power_floor": getattr(args, "relative_power_floor", 1e-4),
                  "selection_metric": result["selection_metric"],
                  "stage_edges": list(STAGE_EDGES), "stage_counts": stage_counts,
                  "component_indices": list(indices),
                  "component_names": list(COMPONENT_NAMES), "component_scale": component_scale,
                  "component_weight": args.component_weight if indices else 0.,
                  "component_scale_scope": f"RMS over {snapshot_scope} training snapshots only; fixed at inference"
                                           + ("; stage weighted" if stage_relative else ""),
                  "component_units": "weighted signed contributions to P21, mK^2 cMpc^3"}
    return checkpoint, result


def predict_checkpoint(checkpoint, case):
    """Predict using saved train-only normalization; accepts a dict or .pt path.

    Field checkpoints return xHI and optional auxiliary fields for all snapshots.
    Direct checkpoints return P21 only. RT labels are not accessed in either path.
    """
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["kind"] == "direct_p21":
        if (case["metadata"]["parameter_names"] != checkpoint["parameter_names"]
                or not np.array_equal(case["redshifts"], checkpoint["redshifts"])
                or case["metadata"]["box_len_cMpc"] != checkpoint["box_len_cMpc"]):
            raise ValueError("Direct P21 checkpoint requires the same parameter definitions, redshifts and box")
        model = direct_model(checkpoint["n_parameters"], checkpoint["n_outputs"], checkpoint["width"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        x = torch.tensor(case["metadata"]["theta"], dtype=torch.float32)
        with torch.no_grad():
            logp = model((x - checkpoint["input_mean"]) / checkpoint["input_scale"])
            logp = logp * checkpoint["output_scale"] + checkpoint["output_mean"]
            power = (logp.exp() - checkpoint["power_floor"]).clamp_min(0).numpy()
        if not np.isfinite(power).all():
            raise FloatingPointError("Nonfinite direct P21 prediction")
        return {"P21": power.reshape(checkpoint["output_shape"])}
    model = FieldNet(len(checkpoint["feature_names"]), width=checkpoint["width"],
                     wide=checkpoint["wide"], auxiliary=checkpoint["auxiliary"],
                     output_transform=checkpoint.get("output_transform", "legacy"),
                     prior_epsilon=checkpoint.get("prior_epsilon", 1e-4))
    if (case["xhi_es"].shape[-1] != checkpoint["grid_dimension"]
            or case["metadata"]["box_len_cMpc"] != checkpoint["box_len_cMpc"]):
        raise ValueError("Field checkpoint requires its training grid and box")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    fields, auxiliaries = [], []
    for snapshot in range(len(case["redshifts"])):
        inputs, names = build_inputs(case, snapshot, history_steps=checkpoint["history_steps"])
        if names != checkpoint["feature_names"]:
            raise ValueError("Prediction feature definitions differ from checkpoint")
        inputs = (torch.from_numpy(inputs[None]) - checkpoint["input_mean"]) / checkpoint["input_scale"]
        baseline = torch.tensor(case["xhi_es"][snapshot][None], dtype=torch.float32)
        with torch.no_grad():
            field, auxiliary = model(inputs, baseline)
        fields.append(field[0].numpy())
        if auxiliary is not None:
            auxiliaries.append(auxiliary[0].numpy())
    result = {"xhi": np.array(fields)}
    if auxiliaries:
        result["auxiliary"] = np.array(auxiliaries)
    return result


def evaluate(checkpoint, cases, output, snapshot_scope="last-three"):
    """Save predictions and independently measure field and power errors."""
    output.mkdir()
    records = []
    for case in cases:
        selected_snapshots = snapshot_mask(len(case["redshifts"]), snapshot_scope)
        prediction = predict_checkpoint(checkpoint, case) if checkpoint is not None else {"xhi": case["xhi_es"]}
        np.savez_compressed(output / f"{case['metadata']['case_id']}.npz", **prediction,
                            redshifts=case["redshifts"])
        for j, z in enumerate(case["redshifts"]):
            amplitude = 27 * np.sqrt((1 + z) / 10)
            rho = density_at(case, j).astype(np.float64)
            factor = amplitude * rho
            fields = {"teacher": factor * case["xhi_rt"][j]}
            if "xhi" in prediction:
                fields["nn"] = factor * prediction["xhi"][j]
            spectra = shell_spectra(fields, case["metadata"]["box_len_cMpc"])
            k = np.array(spectra["k_cMpc_inv"])
            selected = (k >= .1) & (k <= .6 * spectra["k_max_cMpc_inv"])
            if not selected.any():
                raise ValueError("No primary k bins in [0.1, 0.6 kNyquist]")
            truth = np.array(spectra["P_teacher"])
            power = np.array(spectra["P_nn"] if "xhi" in prediction else prediction["P21"][j])
            # Identical reference floor to shell_spectra: undefined bins are counted.
            floor = max(float(truth.max()) * 1e-12,
                        np.finfo(float).eps ** 2 * case["metadata"]["box_len_cMpc"] ** 3
                        * max(float(np.mean(fields["teacher"] ** 2)), 1.))
            errors = [float(abs(p / t - 1)) if t > floor else None for p, t in zip(power, truth, strict=True)]
            row = {"case_id": case["metadata"]["case_id"], "split": case["metadata"]["split"],
                   "snapshot": j, "redshift": float(z), "primary_snapshot": bool(selected_snapshots[j]),
                   "teacher_xHI_mean": float(np.mean(case["xhi_rt"][j])),
                   "k_cMpc_inv": k.tolist(), "primary_k": selected.tolist(), "P_teacher": truth.tolist(),
                   "P_predicted": power.tolist(), "fractional_power_error": errors,
                   "absolute_power_error": np.abs(power-truth).tolist()}
            if checkpoint is not None and checkpoint.get("power_denominator_floor") is not None:
                denominator = np.asarray(checkpoint["power_denominator_floor"], dtype=np.float64)
                if denominator.shape != truth.shape:
                    raise ValueError("Checkpoint spectral floor differs from evaluation shells")
                row["regularized_fractional_power_error"] = (
                    np.abs(power-truth)/np.maximum(truth, denominator)).tolist()
                row["power_denominator_floor"] = denominator.tolist()
            if "xhi" in prediction:
                row["field"] = field_metrics(1 - prediction["xhi"][j], 1 - case["xhi_rt"][j], density_at(case, j))
                row["r21"] = spectra["r_nn"]
                with torch.no_grad():
                    components = [decompose_power(torch.as_tensor(rho, dtype=torch.float64),
                                  torch.as_tensor(value, dtype=torch.float64), amplitude,
                                  case["metadata"]["box_len_cMpc"])["terms"].numpy()
                                  for value in (case["xhi_rt"][j], prediction["xhi"][j])]
                teacher_terms, predicted_terms = components
                for terms, total in zip(components, (truth, power), strict=True):
                    np.testing.assert_allclose(terms.sum(axis=-2), total, rtol=1e-9, atol=1e-10)
                row["P21_terms_teacher"] = teacher_terms.tolist()
                row["P21_terms_predicted"] = predicted_terms.tolist()
                row["component_error_over_P21"] = [
                    [float(delta / t) if t > floor else None
                     for delta, t in zip(difference, truth, strict=True)]
                    for difference in predicted_terms - teacher_terms]
                row["component_closure_relative"] = {
                    name: float(np.max(np.abs(terms.sum(axis=-2) - total))
                                / max(float(np.max(np.abs(total))), 1e-30))
                    for name, terms, total in zip(("teacher", "predicted"), components,
                                                  (truth, power), strict=True)}
            records.append(row)
    write_json(output / "metrics.json", records)
    summary = {}
    for split in sorted({r["split"] for r in records}):
        all_rows = [r for r in records if r["split"] == split]
        rows = [r for r in all_rows if r["primary_snapshot"]]
        summary[split] = summarize_records(rows)
        summary[split]["snapshot_scope"] = snapshot_scope
        if rows and "regularized_fractional_power_error" in rows[0]:
            stage_means = []
            for low, high in itertools.pairwise(STAGE_EDGES):
                stage = [r for r in rows if low <= r["teacher_xHI_mean"]
                         and (r["teacher_xHI_mean"] < high or high == 1.)]
                if stage:
                    stage_means.append(float(np.mean([v for r in stage for v, use in
                        zip(r["regularized_fractional_power_error"], r["primary_k"], strict=True) if use])))
            summary[split]["selection_score"] = float(np.mean(stage_means))
            summary[split]["selection_metric"] = SELECTION_METRIC
        summary[split]["by_snapshot"] = [
            {"snapshot": index, "redshift": next(r["redshift"] for r in all_rows if r["snapshot"] == index),
             "included_in_primary_summary": next(r["primary_snapshot"] for r in all_rows if r["snapshot"] == index),
             **summarize_records([r for r in all_rows if r["snapshot"] == index])}
            for index in sorted({r["snapshot"] for r in all_rows})]
        edges = (0., .01, .1, .5, .9, .99, 1.)
        summary[split]["by_neutral_fraction"] = [
            {"lower": low, "upper": high, "lower_inclusive": True, "upper_inclusive": high == 1.,
             **summarize_records([r for r in rows if low <= r["teacher_xHI_mean"]
                                  and (r["teacher_xHI_mean"] < high or
                                       (high == 1. and r["teacher_xHI_mean"] == 1.))])}
            for low, high in itertools.pairwise(edges)]
    return summary


def summarize_records(rows):
    """Summarize selected snapshots, retaining undefined relative bins and absolute errors."""
    return {"cases": len({r["case_id"] for r in rows}), "snapshots": len(rows),
            "power_fractional_error": distribution(v for r in rows
                for v, use in zip(r["fractional_power_error"], r["primary_k"], strict=True) if use),
            "absolute_power_error": distribution(v for r in rows
                for v, use in zip(r["absolute_power_error"], r["primary_k"], strict=True) if use),
            "absolute_power_error_units": "mK^2 cMpc^3",
            "regularized_power_error": distribution(v for r in rows
                if "regularized_fractional_power_error" in r
                for v, use in zip(r["regularized_fractional_power_error"], r["primary_k"], strict=True) if use),
            "field_MAE": distribution(r["field"]["MAE"] for r in rows if "field" in r),
            "field_correlation": distribution(r["field"]["field_correlation"] for r in rows if "field" in r),
            "r21": distribution(v for r in rows if "r21" in r
                for v, use in zip(r["r21"], r["primary_k"], strict=True) if use),
            "components": {name: distribution(
                abs(value) if value is not None else None
                for row in rows if "component_error_over_P21" in row
                for value, use in zip(row["component_error_over_P21"][index], row["primary_k"], strict=True) if use)
                for index, name in enumerate(COMPONENT_NAMES)}}


def comparison_plot(summary, output):
    """Plot available split errors without selecting models on test data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    available = {split for result in summary.values() for split in result}
    splits = ("test", "external_test") if available & {"test", "external_test"} else ("train", "validation")
    for axis, split in zip(axes, splits, strict=True):
        labels, values = [], []
        for name, results in summary.items():
            metric = results.get(split, {}).get("power_fractional_error", {}).get("mean")
            if metric is not None:
                labels.append(name)
                values.append(100 * metric)
        axis.bar(np.arange(len(labels)), values)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=60, ha="right")
        axis.set(title=split, ylabel="Mean absolute P21 error [%]")
        axis.grid(axis="y", alpha=.2)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def parser():
    """Define CPU pilot controls and explicit dataset/output paths."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(DEFAULT_ARMS))
    p.add_argument("--model-seeds", nargs="+", type=int, default=[11])
    p.add_argument("--snapshot-scope", choices=("last-three", "all"), default="last-three",
                   help="Snapshots for spectral fitting, validation, diagnostics and primary evaluation")
    p.add_argument("--diagnostic-interval", type=int, default=0,
                   help="Record comparable primary-bin train/validation metrics every N epochs; 0 disables")
    p.add_argument("--evaluation-splits", nargs="+", choices=("train", "validation", "test", "external_test"),
                   default=["train", "validation", "test", "external_test"],
                   help="Splits to evaluate after fitting; does not change training or case selection")
    for name, default in (("epochs", 80), ("patience", 15), ("threads", 8), ("batch-size", 8),
                          ("train-cases", 32), ("validation-cases", 8), ("test-cases", 8),
                          ("external-cases", 4), ("history-steps", 3), ("width", 16)):
        p.add_argument(f"--{name}", type=int, default=default)
    p.add_argument("--training-objective", choices=("legacy", "stage-relative"), default="legacy",
                   help="Stage-relative balances teacher stages and uses regularized percentage-power scoring")
    p.add_argument("--output-transform", choices=("legacy", "logit"), default="legacy")
    p.add_argument("--prior-epsilon", type=float, default=1e-4)
    p.add_argument("--relative-power-floor", type=float, default=1e-4,
                   help="Fraction of training-only per-k RMS power used as a denominator floor")
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--power-weight", type=float, default=.01)
    p.add_argument("--auxiliary-weight", type=float, default=.01)
    p.add_argument("--component-weight", type=float, default=.01)
    p.add_argument("--power-floor", type=float, default=1e-8)
    return p


def main(argv=None):
    """Run the requested small ablations and save their complete inventory."""
    args = parser().parse_args(argv)
    for name in ("epochs", "patience", "threads", "batch_size", "train_cases", "validation_cases",
                 "test_cases", "history_steps", "width"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.external_cases < 0 or any(seed < 0 for seed in args.model_seeds):
        raise ValueError("External case count and seeds must be nonnegative")
    if args.diagnostic_interval < 0:
        raise ValueError("Diagnostic interval must be nonnegative")
    if len(set(args.evaluation_splits)) != len(args.evaluation_splits):
        raise ValueError("Evaluation splits must be unique")
    if len(set(args.arms)) != len(args.arms) or len(set(args.model_seeds)) != len(args.model_seeds):
        raise ValueError("Arms and model seeds must be unique")
    for name in ("learning_rate", "power_weight", "auxiliary_weight", "component_weight", "power_floor"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(args.prior_epsilon) or not np.finfo(np.float32).eps <= args.prior_epsilon < .5:
        raise ValueError("prior-epsilon must be at least float32 epsilon and below one half")
    if not np.isfinite(args.relative_power_floor) or args.relative_power_floor <= 0:
        raise ValueError("relative-power-floor must be finite and positive")
    if args.training_objective == "stage-relative" and (args.snapshot_scope != "all" or "direct_p21" in args.arms):
        raise ValueError("stage-relative requires all snapshots and field-model arms")
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    manifest = load_manifest(args.manifest)
    cases = select_cases(manifest, {"train": args.train_cases, "validation": args.validation_cases,
                                   "test": args.test_cases, "external_test": args.external_cases})
    args.output.mkdir(parents=True)
    inventory = [{"case_id": c["metadata"]["case_id"], "split": c["metadata"]["split"],
                  "seed": c["metadata"]["seed"], "sha256": c["sha256"]} for c in cases]
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(args.output / "configuration.json", configuration | {"cases": inventory,
               "manifest_sha256": manifest["source_manifest_sha256"], "torch": str(torch.__version__),
               "convergence_certified": False, "scope": "exploratory existing-label model ablations",
               "selection": (SELECTION_METRIC if args.training_objective == "stage-relative" else
                             "field arms: common primary-bin validation log-P21 MSE; direct arm: validation standardized log-P21 MSE"),
               "primary_metrics": f"{args.snapshot_scope} snapshots, 0.1 <= k <= 0.6 kNyquist; no power floor in valid relative errors",
               "snapshot_weighting": ("equal aggregate weight per occupied teacher xHI stage, independently per split"
                                      if args.training_objective == "stage-relative" else
                                      "equal per snapshot; field and chemistry objectives always use all snapshots"),
               "stage_edges": list(STAGE_EDGES),
               "relative_power_floor_definition": "fraction times stage-weighted training-only per-k RMS P; lower bound power_floor",
               "loss_power_floor_units": "mK^2 cMpc^3; additive log-loss floor or absolute minimum relative-error denominator",
               "direct_p21_scope": "six physical controls to P21; no ionization field output, no IC-seed input",
               "chemistry_supervision": "log10 gas TK plus helium simplex; never inference inputs",
               "component_names": list(COMPONENT_NAMES),
               "component_loss": "signed asinh residual with fixed train-only per-term/per-mode RMS scale",
               "component_groups": {key: list(value) for key, value in COMPONENT_GROUPS.items()},
               "component_loss_scope": f"{args.snapshot_scope} snapshots and primary k bins; weighted contributions to P21",
               "component_comparison_architecture": "all thermo_* arms retain the same local network and auxiliary head"})
    evaluation_cases = [case for case in cases if case["metadata"]["split"] in args.evaluation_splits]
    summary = {"es": evaluate(None, evaluation_cases, args.output / "es", args.snapshot_scope)}
    write_json(args.output / "summary.json", summary)
    for seed in args.model_seeds:
        for arm in args.arms:
            name = f"{arm}_seed{seed}"
            print(json.dumps({"starting": name}), flush=True)  # noqa: T201
            random.seed(seed)
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True)
            folder = args.output / name
            folder.mkdir()
            checkpoint, training = fit_direct(cases, args, seed) if arm == "direct_p21" else fit_field(cases, arm, args, seed)
            checkpoint.update(arm=arm, seed=seed, convergence_certified=False)
            torch.save(checkpoint, folder / "model.pt")
            write_json(folder / "training.json", training)
            summary[name] = evaluate(checkpoint, evaluation_cases, folder / "predictions", args.snapshot_scope)
            write_json(args.output / "summary.json", summary)
            print(json.dumps({"completed": name, "test": summary[name].get("test")}), flush=True)  # noqa: T201
    comparison_plot(summary, args.output)
    (args.output / "COMPLETED").write_text("Small model comparison completed; inspect summary.json\n")


if __name__ == "__main__":
    main()
