# Reproduce the geometry-refinement campaign

This workflow preserves the native 21cmFAST source/atom inputs and analytic
teacher physics. It adds numerical input normalization, box geometry inputs and
mixed-geometry training. It does not add recombinations, thermal evolution,
photoheating feedback or independent helium chemistry.

Generation uses the recorded native cumulative photon arrays, initially neutral
coeval gas, `source_periodic_cube` boundaries, 96 directions and 32 exposure
steps. The supplied native input template and compiled native extension are
hashed and checked against the reused campaigns. Native initial conditions use
DIM=96 throughout. All new profiles use redshifts
`[10, 9, 8, 7.5, 7, 6.5, 6]`.

| Profile | Seeds | Box/cells | Snapshots |
|---|---|---|---:|
| Additional resolution training | 2241–2244 | L50/N32 | 28 |
| Additional volume training | 2251–2254 | L256/N32 | 28 |
| Resolution validation | 2261–2262 | L50/N32 | 14 |
| Volume validation | 2271–2272 | L256/N32 | 14 |
| New original-geometry blind test | 2201–2202 | L50/N16 | 14 |
| New resolution blind test | 2211–2212 | L50/N32 | 14 |
| New volume blind test | 2221–2222 | L256/N32 | 14 |
| New unseen-box blind test | 2231–2232 | L100/N32 | 14 |

The single-geometry training set contains 178 snapshots: seeds 2101–2112,
2115–2116 and 2121–2128. Mixed training adds 72 snapshots from previously inspected
seeds 2131–2132/2141–2142 and the 56 new training snapshots. Both candidates use
the same 42 validation snapshots: seeds 2113–2114,2261–2262,2271–2272, with 14
snapshots at each of L50/N16, L50/N32 and L256/N32. Previously inspected
simulations are development data; the new blind seeds do not enter either
collection or model selection.

The two JSON templates specify the reproducible example training settings. Both use
2 workers × 1 thread, model seeds 23/83, 150 initial epochs, 600 budget epochs,
60 residual epochs, 1024 sampled cells per snapshot, 89 field inputs, a 0.01
feature-scale floor and zero weight on constant features. Frame caches are
1536 MiB per worker for the single-geometry control and 4096 MiB per worker for
mixed training. Generation uses 2 workers × 2 threads. Training candidates run
sequentially.

## Existing inputs

`--native-campaign-root` must contain:

- `teacher_results_verified` (seeds 2101–2108) and
  `teacher_results_external` (2109–2112); or the portable generator names
  `teacher_development_original` and `teacher_development_external`.

`--previous-campaign-root` must contain:

- `teacher_train` or `teacher_train_new` (2121–2128);
- `teacher_validation` (2113–2116);
- `teacher_transfer_resolution` (2131–2132);
- `teacher_transfer_volume` (2141–2142);
- the previous selected checkpoint in `selected_model/`.

The checkpoint lookup also accepts `model/`, then
`train_local_features/model/`, in that order. Campaign aliases must resolve to
one archive each. Collections retain original receipts and point to the local
archive paths; copying a collection JSON alone does not relocate its data.

When generating additional teachers, use the same native runtime as the reused
archives. If its compiled extension differs from the archived runtime, regenerate the
earlier campaigns using
`completion_20261116/reproduction/generate_teachers.py` in that environment.
The existing archives are never relabeled to hide a runtime mismatch.

## Prepare and run

Use a committed checkout of the repository. If the teacher archives have
already been downloaded, start with the archive-only route below. Native
simulation is unnecessary for retraining on completed arrays.

### Existing or relocated teacher archives

Place the eight completed `teacher_<profile>/` directories from the table in
one generated-campaign directory. Keep the earlier native and previous-campaign
directories described above, including their original receipts and checkpoint.
Paths inside collection manifests are rebuilt locally; teacher arrays and
receipts remain unchanged.

```bash
pc_python=/path/to/training/python
pc_previous=/path/to/previous_completed_campaign
pc_native=/path/to/earlier_native_campaign
pc_generated=/path/to/completed_refinement_teachers
pc_output=/path/to/new_training_run
pc_module=research.pc_analytic_emulator.refinement_20261117.reproduction.run_campaign

"$pc_python" -m "$pc_module" prepare-training \
  --previous-campaign-root "$pc_previous" \
  --native-campaign-root "$pc_native" \
  --generated-campaign-root "$pc_generated" \
  --output-root "$pc_output"

"$pc_python" -m "$pc_module" train --output-root "$pc_output"
"$pc_python" -m "$pc_module" freeze --output-root "$pc_output"
```

This mode checks that all teacher archives share the same recorded native
runtime, teacher implementation and physical settings. It does not import or
execute that native extension. NumPy, SciPy and Matplotlib suffice for the
training environment; its versions and interpreter are frozen separately.
All new blind archives remain excluded from the development collections.

### Generate new native simulations

To reproduce the new simulations as well, use the supported native environment:

```bash
pc_python=/path/to/python
pc_native_python=/path/to/native/environment/bin/python
pc_previous=/path/to/previous_completed_campaign
pc_native=/path/to/earlier_native_campaign
pc_output=/path/to/new_refinement_campaign
pc_module=research.pc_analytic_emulator.refinement_20261117.reproduction.run_campaign

"$pc_python" -m "$pc_module" prepare \
  --previous-campaign-root "$pc_previous" \
  --native-campaign-root "$pc_native" \
  --output-root "$pc_output" \
  --native-python "$pc_native_python" \
  --native-template research/pc_analytic_emulator/completion_20261116/reproduction/native_template.json

"$pc_python" -m "$pc_module" generate --output-root "$pc_output"
"$pc_python" -m "$pc_module" train --output-root "$pc_output"
"$pc_python" -m "$pc_module" freeze --output-root "$pc_output"
```

Both preparation routes create a detached source worktree and freeze both configurations,
the native/runtime identities and the selection rule before fitting. Later
stages re-enter this frozen source. Native Python symlinks are retained so the
intended virtual environment remains active.

`generate` builds 140 new teacher snapshots and the two audited development
collections. It reports only completeness and photon-ledger checks for the
new blind data. `train` fits the two candidates with empty test splits.
`freeze` scores both completed candidates and the unchanged previous checkpoint
with the identical projected validation objective. The previous model remains
eligible and wins exact ties.

The selected model is copied into `selected_model/`; `selection.json` records
all candidate paths/hashes, 42-row validation evidence, source identities and
the freeze time. A failed stage can resume with the same command. Completed
training is audited on resume; after selection exists, training becomes an
audit and performs no fitting.

## Explicit post-freeze evaluation

New blind prediction metrics are computed only by this separate command:

```bash
"$pc_python" -m "$pc_module" evaluate \
  --output-root "$pc_output" \
  --freeze-record "$pc_output/selection.json"
```

The command verifies the exact selected weights, all validation records and
the source/runtime before opening blind arrays. It benchmarks all four new
profiles with 4096 MFP rays per phase. Interrupted benchmark directories are
preserved and retried with the same frozen weights. L100/N32 remains an unseen
geometry even if the mixed model wins; if the previous model wins, its training
support is retained.

To freeze already-completed runs independently of the staged launcher:

```bash
"$pc_python" -m research.pc_analytic_emulator.refinement_20261117.reproduction.freeze_completed \
  --single-run "$pc_output/train_guarded_single" \
  --mixed-run "$pc_output/train_guarded_mixed" \
  --previous-model "$pc_previous/selected_model" \
  --validation-config "$pc_output/guarded_mixed.json" \
  --training-protocol "$pc_output/training_protocol.json" \
  --output-root "$pc_output"
```

For those independently frozen runs, the matching evaluator does not require
the staged preparation manifest:

```bash
"$pc_python" -m research.pc_analytic_emulator.refinement_20261117.reproduction.evaluate_completed \
  --campaign-root "$pc_output" \
  --selection "$pc_output/selection.json"
```

Add `--teacher-root "$pc_generated"` if the four completed `teacher_blind_*`
directories are stored outside the output root. This evaluator writes the same
immutable gate and protocol, audits all four archives before computing any
predictions, and never retrains or selects another checkpoint.

## Completed-result summaries and oracle diagnostics

After all four benchmarks finish, generate the compact public summary:

```bash
"$pc_python" -m research.pc_analytic_emulator.refinement_20261117.reproduction.summarize_results \
  --root "$pc_output"
```

This writes `scientific_summary.json` after checking completion receipts,
the pre-evaluation gate, and frozen model, source and input identities. It
retains all, active and late snapshots, including undefined power ratios and
empty MFP phases.

Posthoc oracle controls use teacher labels to investigate remaining errors.
They hold the field weights fixed and replace the final absorption count,
replace every count in the frozen base chain, or project the teacher field to
the predicted count. These controls are unavailable during deployment. Their
effects are not an additive error partition; a capacity-L2 projection of the
teacher gives a weighted-MSE lower bound only at the fixed predicted count.
No analogous bound applies to power-spectrum error.

To generate the four diagnostic directories after the benchmarks complete:

```bash
pc_teacher_root="$pc_output"  # Set to the relocated teacher root if required.
for pc_profile in original resolution volume unseen_box; do
  case "$pc_profile" in
    original) pc_seeds=(2201 2202) ;;
    resolution) pc_seeds=(2211 2212) ;;
    volume) pc_seeds=(2221 2222) ;;
    unseen_box) pc_seeds=(2231 2232) ;;
  esac
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$pc_python" -m research.pc_analytic_emulator.diagnose_errors \
    --model "$pc_output/selected_model" \
    --teacher "$pc_teacher_root/teacher_blind_$pc_profile" \
    --output "$pc_output/oracle_blind_$pc_profile" \
    --seeds "${pc_seeds[@]}"
done
```

The diagnostic CLI requires new output directories. Existing saved controls
can be verified directly with the portable evidence builder:

```bash
"$pc_python" -m research.pc_analytic_emulator.refinement_20261117.reproduction.oracle_evidence \
  --root "$pc_output"
```

It writes `oracle_evidence_portable.json`, checks every saved deployed field
against the corresponding frozen benchmark, and verifies unchanged model/source
hashes, oracle mass targets and the applicable weighted-MSE bounds. It performs
no inference or fitting. When `oracle_evidence.json` exists, its scientific
payload must agree exactly; the new report records both the original builder
hash and the portable builder hash. The original evidence remains unchanged.
Use `--original-evidence PATH` to supply another original record and `--output`
to choose a different summary destination. Neither public summary contains
machine-specific absolute paths.

## Gadi

Run preparation on the login node; submit generation/training to PBS. The
supplied script requests 4 CPUs, 32 GB and 48 h. Load the same modules used for
the chosen training environment and, when generating, the native runtime. An optional
`PC_RUNTIME_ENV` file can define `PC_MODULES=(...)`; module names are not guessed.

```bash
qsub -P YOUR_PROJECT \
  -l storage=gdata/YOUR_PROJECT+scratch/YOUR_PROJECT \
  -v PC_RUN="$pc_output",PC_PYTHON="$pc_python",PC_STAGE=development \
  "$pc_output/source/research/pc_analytic_emulator/refinement_20261117/reproduction/gadi_campaign.pbs"
```

For downloaded archives prepared with `prepare-training`, use
`PC_STAGE=train-freeze` instead of `development`; this runs no native simulations.
These are prepared Gadi commands, not an execution record for a remote run.

The job stops after the selection freeze. Submit evaluation separately once
the freeze record exists:

```bash
qsub -P YOUR_PROJECT \
  -l storage=gdata/YOUR_PROJECT+scratch/YOUR_PROJECT \
  -v PC_RUN="$pc_output",PC_PYTHON="$pc_python",PC_STAGE=evaluate,PC_FREEZE_RECORD="$pc_output/selection.json" \
  "$pc_output/source/research/pc_analytic_emulator/refinement_20261117/reproduction/gadi_campaign.pbs"
```

Generation additionally needs the supported native 21cmFAST extension and the
repository's native build dependencies. No package installation or runtime
replacement is performed by these scripts.
