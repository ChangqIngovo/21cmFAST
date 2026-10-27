"""CPU-sized history/chemistry/context/P21 ablations on existing pilot fields.

No native simulations run here. All splits are whole histories. RT thermal and
helium fields are auxiliary labels, never deployment inputs. Checkpoints retain
train-only normalization and expose ``predict_checkpoint`` for field inference.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from research.es_pc_emulator.physical_pilot_training import load_case, load_manifest
from research.es_pc_emulator.small_pilot_model import (
    FieldNet,
    build_inputs,
    power_loss,
    shell_power,
    thermal_targets,
)
from research.pc_field_emulator.evaluation import field_metrics, shell_spectra

ARMS = ("local", "history", "thermo", "spatial", "p21", "direct_p21")


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


def field_arrays(cases, history_steps, auxiliary):
    """Build full-cube batches with separate deployment inputs and labels."""
    rows = {name: [] for name in ("inputs", "base", "truth", "brightness", "aux", "primary")}
    names = None
    for case in cases:
        for snapshot, redshift in enumerate(case["redshifts"]):
            inputs, columns = build_inputs(case, snapshot, history_steps=history_steps)
            if names is not None and names != columns:
                raise ValueError("Feature columns changed between cases")
            names = columns
            rows["inputs"].append(inputs)
            rows["base"].append(case["xhi_es"][snapshot])
            rows["truth"].append(case["xhi_rt"][snapshot])
            rows["primary"].append(snapshot >= len(case["redshifts"]) - 3)
            rows["brightness"].append(27 * np.sqrt((1 + redshift) / 10) * density_at(case, snapshot))
            if auxiliary:
                rows["aux"].append(thermal_targets(case, snapshot))
    arrays = {key: torch.from_numpy(np.asarray(value, dtype=np.float32))
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


def field_loss(model, batch, power_weight, auxiliary_weight, power_floor, box):
    """Combine field error with enabled chemistry and spectral objectives."""
    prediction, auxiliary = model(batch["inputs"], batch["base"])
    loss = torch.mean((prediction - batch["truth"]) ** 2)
    if auxiliary is not None:
        # Temperature is log10(K); helium fractions have a simplex constraint.
        loss = loss + auxiliary_weight * torch.mean((auxiliary - batch["aux"]) ** 2)
    if power_weight:
        _, predicted, _ = shell_power(batch["brightness"] * prediction, box)
        _, truth, _ = shell_power(batch["brightness"] * batch["truth"], box)
        loss = loss + power_weight * power_loss(predicted, truth, floor=power_floor)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training/validation loss")
    return loss


def train_loop(model, train, validation, loss_function, args, seed, validation_function=None):
    """Optimize batches and retain the best validation-selected parameters."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    best, best_state, stale = float("inf"), None, 0
    history = []
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
        if val_loss < best:
            best, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
            best_epoch = epoch + 1
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % 10 == 0 or stale >= args.patience:
            print(json.dumps(history[-1]), flush=True)  # noqa: T201
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return {"history": history, "best_epoch": best_epoch, "best_validation_loss": best}


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
    auxiliary = arm in ("thermo", "spatial", "p21")
    wide = arm in ("spatial", "p21")
    train, names = field_arrays([c for c in cases if c["metadata"]["split"] == "train"], history_steps, auxiliary)
    validation, validation_names = field_arrays(
        [c for c in cases if c["metadata"]["split"] == "validation"], history_steps, auxiliary)
    if names != validation_names:
        raise ValueError("Training and validation features differ")
    mean, scale = normalization(train["inputs"])
    train["inputs"] = (train["inputs"] - mean) / scale
    validation["inputs"] = (validation["inputs"] - mean) / scale
    model = FieldNet(len(names), width=args.width, wide=wide, auxiliary=auxiliary)
    box = cases[0]["metadata"]["box_len_cMpc"]
    weight = args.power_weight if arm == "p21" else 0.

    def validation_score(network, batch):
        # All field arms select epochs on the SAME observable and physical bins.
        selected = {key: value[batch["primary"].bool()] for key, value in batch.items()}
        predicted, _ = network(selected["inputs"], selected["base"])
        k, pp, _ = shell_power(selected["brightness"] * predicted, box)
        _, pt, _ = shell_power(selected["brightness"] * selected["truth"], box)
        mask = (k >= .1) & (k <= .6 * np.pi * predicted.shape[-1] / box)
        if not mask.any():
            raise ValueError("No primary validation k bins")
        return power_loss(pp[..., mask], pt[..., mask], floor=args.power_floor)

    result = train_loop(model, train, validation,
                        lambda m, b: field_loss(m, b, weight, args.auxiliary_weight, args.power_floor, box),
                        args, seed, validation_function=validation_score)
    checkpoint = {"kind": "field", "state_dict": model.state_dict(), "input_mean": mean, "input_scale": scale,
                  "feature_names": names, "history_steps": history_steps, "width": args.width,
                  "wide": wide, "auxiliary": auxiliary, "box_len_cMpc": box,
                  "grid_dimension": cases[0]["xhi_es"].shape[-1],
                  "power_weight": weight, "auxiliary_weight": args.auxiliary_weight, "power_floor": args.power_floor}
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
                     wide=checkpoint["wide"], auxiliary=checkpoint["auxiliary"])
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


def evaluate(checkpoint, cases, output):
    """Save predictions and independently measure field and power errors."""
    output.mkdir()
    records = []
    for case in cases:
        prediction = predict_checkpoint(checkpoint, case) if checkpoint is not None else {"xhi": case["xhi_es"]}
        np.savez_compressed(output / f"{case['metadata']['case_id']}.npz", **prediction,
                            redshifts=case["redshifts"])
        for j, z in enumerate(case["redshifts"]):
            factor = 27 * np.sqrt((1 + z) / 10) * density_at(case, j)
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
                   "snapshot": j, "redshift": float(z), "primary_snapshot": j >= len(case["redshifts"]) - 3,
                   "k_cMpc_inv": k.tolist(), "primary_k": selected.tolist(), "P_teacher": truth.tolist(),
                   "P_predicted": power.tolist(), "fractional_power_error": errors}
            if "xhi" in prediction:
                row["field"] = field_metrics(1 - prediction["xhi"][j], 1 - case["xhi_rt"][j], density_at(case, j))
                row["r21"] = spectra["r_nn"]
            records.append(row)
    write_json(output / "metrics.json", records)
    summary = {}
    for split in sorted({r["split"] for r in records}):
        rows = [r for r in records if r["split"] == split and r["primary_snapshot"]]
        summary[split] = {"cases": len({r["case_id"] for r in rows}),
                          "power_fractional_error": distribution(v for r in rows
                              for v, use in zip(r["fractional_power_error"], r["primary_k"], strict=True) if use),
                          "field_MAE": distribution(r["field"]["MAE"] for r in rows if "field" in r),
                          "field_correlation": distribution(r["field"]["field_correlation"] for r in rows if "field" in r),
                          "r21": distribution(v for r in rows if "r21" in r
                              for v, use in zip(r["r21"], r["primary_k"], strict=True) if use)}
    return summary


def comparison_plot(summary, output):
    """Plot held-out power errors without selecting models on test data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, split in zip(axes, ("test", "external_test"), strict=True):
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
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--model-seeds", nargs="+", type=int, default=[11])
    for name, default in (("epochs", 80), ("patience", 15), ("threads", 8), ("batch-size", 8),
                          ("train-cases", 32), ("validation-cases", 8), ("test-cases", 8),
                          ("external-cases", 4), ("history-steps", 3), ("width", 16)):
        p.add_argument(f"--{name}", type=int, default=default)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--power-weight", type=float, default=.01)
    p.add_argument("--auxiliary-weight", type=float, default=.01)
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
    if len(set(args.arms)) != len(args.arms) or len(set(args.model_seeds)) != len(args.model_seeds):
        raise ValueError("Arms and model seeds must be unique")
    for name in ("learning_rate", "power_weight", "auxiliary_weight", "power_floor"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
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
               "selection": "field arms: common primary-bin validation log-P21 MSE; direct arm: validation standardized log-P21 MSE",
               "primary_metrics": "last three snapshots, 0.1 <= k <= 0.6 kNyquist; no power floor in valid relative errors",
               "loss_power_floor_units": "mK^2 cMpc^3; used only in log-loss and direct-output transform",
               "direct_p21_scope": "six physical controls to P21; no ionization field output, no IC-seed input",
               "chemistry_supervision": "log10 gas TK plus helium simplex; never inference inputs"})
    summary = {"es": evaluate(None, cases, args.output / "es")}
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
            summary[name] = evaluate(checkpoint, cases, folder / "predictions")
            write_json(args.output / "summary.json", summary)
            print(json.dumps({"completed": name, "test": summary[name].get("test")}), flush=True)  # noqa: T201
    comparison_plot(summary, args.output)
    (args.output / "COMPLETED").write_text("Small model comparison completed; inspect summary.json\n")


if __name__ == "__main__":
    main()
