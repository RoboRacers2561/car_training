# Models, lifecycle, and independent review handoff

Historical record only: commands below are unsupported. Use `uv run rc-car` and
the [current README](../../README.md). These transcripts document earlier checks.

Updated 2026-09-25 when Joshua requested a pause to conserve tokens. This is a documentation-only closeout. Do not resume implementation or allocate cloud resources until the user resumes the work.

## Scope and coordination

- Working branch: `astra-cleanup`.
- This agent implemented portable models and lifecycle tests, then independently reviewed the numerical core, Isaac adapters, telemetry, CLI, provenance, and runner recovery.
- Root owns package metadata, dependency locks, CLI, telemetry/lifecycle implementation, runner changes, documentation, and CI. The simulator agent owns numerical core and Isaac math adapters. The tooling agent owns experiment configuration/tracking.
- There were substantial pre-existing uncommitted and untracked changes. Preserve them. This agent made no commits, created no cloud resources, launched no Isaac application, and sent no external messages.
- Root saved a verified baseline archive before legacy code changes at `outputs/refactor-baseline-20260925/source-before-refactor.tar.gz`.
- The latest root change described below has not received this agent's independent re-review. Do not describe it as verified by the earlier test results.

## Portable model changes

Files implemented by this agent:

- `src/rc_car/models/__init__.py`
- `src/rc_car/models/feature_extractor.py`
- `src/rc_car/models/inputs.py`
- `tests/models/test_feature_extractor.py`
- `tests/models/fixtures/legacy_feature_extractor.py`
- `source/car_training/car_training/tasks/manager_based/car_training/agents/feature_extractor.py`

`ResidualBlock` and `CustomMultiInputExtractor` moved into the portable package. The old feature-extractor leaf module re-exports those classes so historical dotted class references can resolve when the legacy Isaac package and the new package are installed. Root wired `rc-car-research[learning]` into the Isaac environment and legacy extension installation requirements.

Important preserved behavior:

- Same layer names, architecture, state-dict keys, initialization order, and random-number consumption.
- Same residual in-place addition and activation order.
- Same constructor prints.
- Same train-mode dummy CNN forward used to determine the flattened size. This updates BatchNorm statistics during construction; counters deliberately start at one. This is inherited behavior, not a new design recommendation.
- No normalization, clipping, or input validation was inserted into the training forward pass.
- Exact old/new initial state, train/eval outputs, gradients, and updated BatchNorm buffers were compared against a frozen historical implementation.

The common historical observation contract is float32 track images `(B, 1, 48, 48)`, obstacle history `(B, 4, 48, 48)`, and velocity `(B, 2)`. One obstacle channel and non-square compatible images are also exercised. Images produced by the observation functions are normalized to `[0, 1]`; their environment metadata has historically allowed `[0, 255]`. Velocity normalization divides by `0.8 m/s` without clipping.

`validate_feature_extractor_inputs` is an explicit preflight/analysis diagnostic. It checks declared shapes, channel-first image layout, common batch/device, nonempty batches, float32, finiteness, shared spatial dimensions, and image range. Velocity values beyond `[-1, 1]` remain allowed. The validator does not modify inputs and is not automatically called by `forward`. Value checks synchronize CUDA data, so do not add it unconditionally to every training step.

The portable `rc_car.models` import does not import Isaac or `car_training`. The legacy enclosing `car_training` package still eagerly imports Isaac: the shim does not make that old package portable.

## Model verification

Last independently verified result: **19 model tests passed in 3.35 seconds**.

The suite covers:

- Direct isolated loading of the historical fixture without executing its Isaac parent imports.
- Exact state/output/input-gradient/parameter-gradient parity in both train and eval modes.
- Valid shapes, normalized ranges, input immutability, and useful contract failures.
- BatchNorm constructor and train/eval state changes.
- Strict `state_dict` round trip.
- One tiny CPU PPO update with finite, changed model parameters.
- SB3 checkpoint save/reload with identical deterministic actions.
- A checkpoint written with the old dotted extractor module reference, then loaded through the compatibility shim. Fake parent namespaces are used only to avoid starting Isaac; this is not a full extension-import test.
- Fresh-process portable import isolation.

Commands used after root synchronized dependencies:

```sh
MPLCONFIGDIR=/private/tmp/rc-car-mpl-model-tests .venv/bin/pytest tests/models/test_feature_extractor.py -q
.venv/bin/ruff check src/rc_car/models tests/models
.venv/bin/ty check src/rc_car/models tests/models --output-format concise
```

Ruff and ty passed for this scope. The frozen legacy fixture is excluded from formatting and typing, intentionally preserving historical source. The tooling agent independently confirmed its AST matches the original Git implementation apart from its added fixture docstring, and found no model blockers. That agent also reran the 19 tests and scoped typing checks successfully.

Known model limits:

- A `[0, 1]` range check cannot detect accidental extra division by 255; tiny normalized values still satisfy the range. Observation preprocessing parity needs recorded real samples later.
- CPU synthetic PPO verifies software wiring, not learned driving, rendering, physics, or GPU throughput.
- Historical checkpoint import compatibility is tested with an isolated namespace, not a running Isaac extension.
- Changes to BatchNorm, normalization, architecture, or actor input conventions should be separately versioned and evaluated, not folded into extraction cleanup.

## Independent core and Isaac adapter review

Reviewed `src/rc_car/core/{geometry,actions,rewards,_validation}.py`, the associated tests, and changes to legacy `mdp/actions.py`, `mdp/observation.py`, and `mdp/rewards.py`.

Independent result: **29 core and adapter-body tests passed**. No blocking semantic, shape, device, or unit regression was found. A separate in-memory AST comparison against the baseline archive checked coordinate transforms in eight combinations of float32/float64, aspect preservation, and forward/inverse direction; all were bit-exact.

Preserved conventions include:

- World-edge mapping to image width/height, not the final pixel center.
- Float32 coordinate output by default.
- Historical actor velocity rotation `R(-yaw - pi/2)`.
- Tanh action mapping, nonzero driving minimum, wheel-speed and steering units.
- Unclipped contour-distance reward, obstacle gating, and steering-magnitude penalty.

The deliberate behavior fix is `robot_vel` returning an independent tensor instead of modifying the simulator's cached `root_com_lin_vel_w` through a slice. Root records this in the run manifest under a new behavior version. Do not promise full historical trajectory parity after that fix.

One documentation correction was raised and resolved: the adapter's tanh is not necessarily a "second tanh." The supported unsquashed SB3 PPO clips Gaussian actions to the Box bounds before the adapter applies tanh. An input of ±1 reaches approximately ±0.456956 radians for a nominal 0.6-radian steering scale, and about 12.384–27.616 rad/s for a nominal 10–30 rad/s wheel-speed interval. Core golden tests include those values.

AST-isolated adapter tests execute actual function bodies and their actual portable imports with fake state. They cannot verify Isaac imports, manager ordering, simulator stepping, camera output, or reset lifecycle. Observation history side effects remain deferred because fixing them requires an explicit once-per-step integration point.

## Telemetry and CLI review

Reviewed `src/rc_car/telemetry.py`, `tests/test_telemetry.py`, `src/rc_car/cli.py`, `tests/test_cli.py`, and the runner diff against the baseline archive.

Independent result before later lifecycle edits: **18 telemetry/CLI tests passed in 1.35 seconds**, and scoped ty checks passed.

Verified directly against the pinned local Isaac Lab source:

- `reward_manager.py` computes each weighted contribution including dt, then stores `value / dt` in `_step_reward`.
- Reward-manager reset does not clear `_step_reward`, so the post-step callback can read the just-completed transition's term values.
- `RewardRollup` therefore correctly treats `_step_reward` as weighted before dt and applies dt once when accumulating contributions. The float64 reconstruction is a numerical summary, not a promise of bit-identical reproduction of the manager's float32 summation.
- The new reward accumulator avoids per-step CPU synchronization; its snapshot transfers accumulated data. Termination counting and video capture still synchronize, as the inherited implementation did.
- SB3 2.7 calls the step callback before timeout bootstrap modifications. Recorded environment reward and PPO learning-target reward must remain distinguished.
- A callback returning false stops collection without optimizing that partial rollout. SB3 calls training-end hooks for this normal stop, but not when collection or optimization raises.
- `_on_rollout_end` occurs before optimization of that rollout; checkpoint/event labels must retain that distinction.

Important pinned-manager limitation: Isaac Lab's `TerminationManager.compute()` clears all `_term_dones` columns for each matching row before setting the current cause. If multiple causes are true, `get_term()` exposes only the last matching configured cause. The general portable counting helper supports overlapping masks, but the current adapter cannot reconstruct overlaps already discarded by the manager. Root added a manifest note describing this. Do not rerun stateful termination functions from the callback to obtain extra causes.

## Lifecycle findings, fixes, and tests

This agent added `tests/test_training_lifecycle.py`. Root owns `src/rc_car/lifecycle.py` and the corresponding runner changes.

Findings raised and fixed by root:

1. Partial rollout telemetry was only flushed by `_on_training_end`, so exceptions bypassed it. `Metrics.flush_partial_rollout(reason=...)` now supports both normal and exception cleanup, marks samples `optimized=False`, and clears only after a successful event write.
2. A video or upload failure could prevent the interrupted checkpoint attempt or replace the original learning error. The exception path now attempts the checkpoint first, then partial telemetry, then video closure using `best_effort_cleanup`.
3. The outer `main()` finally block could let W&B finish errors replace the preserved learner error and skip `env.close()`. Root added `finish_resources`, preserving an active error or propagating the first cleanup error while attempting remaining resources.

The tests compile actual `Metrics`, `reset_counts`, the `train()` learning try/except block, and the `main()` try/finally block from the runner AST. These tests exercise real integration ordering rather than copies of the cleanup logic, without importing Isaac.

Last independently verified result before the latest root edit: **15 lifecycle tests passed in 1.20 seconds**, with scoped Ruff, ty, and formatting checks passing. `git diff --check` was also clean at that point.

Coverage at that verified point:

- Budget-stop partial event, timestep counts, dt contribution, and `optimized=False`.
- Repeated cleanup does not duplicate a successful partial event.
- Video-close failure still allows valid partial telemetry to flush.
- Checkpoint attempted before other recovery operations.
- Checkpoint, nonfinite-summary, and media failures cannot replace the original learner exception or prevent later cleanup attempts.
- Recovery notes include exception classes rather than potentially sensitive cleanup exception messages.
- Interrupt handling still attempts later cleanup operations.
- Eight combinations of learner, tracking-finish, and environment-close success/failure verify exception identity, exit-code reporting, and environment-close attempts.

Commands used:

```sh
MPLCONFIGDIR=/private/tmp/rc-car-mpl-model-tests .venv/bin/pytest tests/test_training_lifecycle.py -q
.venv/bin/ruff check tests/test_training_lifecycle.py
.venv/bin/ty check tests/test_training_lifecycle.py --output-format concise
.venv/bin/ruff format --check tests/test_training_lifecycle.py src/rc_car/lifecycle.py
```

### Latest root edit: pending independent re-review

Immediately before the pause, root changed `Metrics._on_training_end` to use `finish_resources` too. This closes the edge case where video closure fails and partial-summary flushing also fails: the first media error should survive instead of being replaced by the summary error.

Root also added `finish_resources` to the lifecycle test fixture namespace and parametrized the video-close failure test with `invalid_terms`. Root reported a targeted run in progress under execution session `21414`.

This agent was explicitly told not to poll, rerun tests, or edit implementation after the pause. The outcome of that latest run is not known here. The prior 15-test result does not verify this final change, and the current test count may differ.

## Provenance review

New inclusion of `src/`, `.github/`, `.yml`, `.typed`, and root tooling configuration is appropriate. No symlinks or W&B token-marker matches were found in the selected source trees during this review. This was a narrow check, not a comprehensive secret audit.

Limitations retained:

- The archive is a source snapshot. It excludes binary assets such as PNG, PT, and USD files, so it is not a complete runtime backup.
- The generated fixture does not need external meshes/textures; historical tasks still need separately preserved asset dependencies.
- `.isaaclab` is an editable dependency and is not included in the source hash list. `setup_isaac.sh` pins its revision and rejects a dirty checkout, but recording actual revision and dirty status in each runtime manifest would make provenance stronger.
- At review time `.isaaclab` was clean at `3c6e67bb5c7ada942a6d1884ab69338f57596f77` (v2.3.0). Recheck before future runtime work.

## Dependencies and tool coordination

Root synchronized a Python 3.11 environment containing torch 2.7.0, Stable-Baselines3 2.7.0, Gymnasium, pytest 9.1.1, ty 0.0.84, and Ruff 0.16.9. Hypothesis is used by numerical core tests. Torch/SB3 are in the optional `learning` extra; configuration inspection should not require them.

Use the existing `.venv/bin` tools for independent checks while root is editing locks; do not race simultaneous `uv sync` processes. Root's intended full check command is:

```sh
uv run --locked --extra learning --group tracking python scripts/check.py
```

The temporary `MPLCONFIGDIR` used above avoids attempting to create a Matplotlib cache in the restricted home directory. No credential values belong in tests, documentation, or manifests.

## Exact next steps after the user resumes

1. Ask root for the result of the last targeted run (`21414`), or inspect recorded results if the execution session no longer exists. Do not assume success.
2. Read the latest `Metrics._on_training_end`, `finish_resources`, and changed lifecycle test. Independently verify the simultaneous media-error/nonfinite-summary case retains the first error while attempting both cleanups.
3. Run the targeted lifecycle suite and scoped Ruff/ty checks only if needed to complete verification of that last edit. Report the actual resulting test count.
4. Root should run the agreed full local quality gate once all agent changes are settled, then inspect the combined diff and preserve the baseline and user changes.
5. Keep the difference between portable CPU checks, isolated adapter-body tests, and unperformed Isaac/GPU integration checks explicit in the handoff.
6. Do not start cloud compute, publish artifacts, or continue larger refactor phases merely because this handoff exists. Resume only under the user's renewed direction.
