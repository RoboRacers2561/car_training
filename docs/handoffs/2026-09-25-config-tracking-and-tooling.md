# Configuration, tracking, tooling, and model-review handoff

Historical record only: commands and environment layouts below are unsupported.
Use `uv run rc-car` and the [current README](../../README.md).

Date: 2026-09-25

Status: paused at the user's request to conserve tokens. This document records
completed work and verification; it does not authorize another run, cloud
allocation, installation, or implementation step. Resume only when the user asks.

All implementation described below was performed on `astra-cleanup`. The root
agent owns the overall branch, baseline preservation, packaging, documentation,
and integration. No commits were made by this subagent. Existing recovery work
and generated artifacts were preserved.

## Owned implementation files

- `src/rc_car/experiments/config.py`: strict portable recipe models and loading.
- `src/rc_car/experiments/tracking.py`: optional W&B integration and run identity.
- `src/rc_car/experiments/__init__.py`: public exports.
- `scripts/experiment_tracking.py`: compatibility imports for historical scripts.
- `tests/experiments/test_config.py`: recipe and import-boundary regressions.
- `tests/experiments/test_tracking.py`: fake-SDK tracking regressions.
- `tests/test_experiment_tracking.py`: the five existing unittest checks, adapted
  to import the compatibility shim and mock the optional SDK without installing it.

The root agent owns `pyproject.toml`, lockfiles, `rc_car/__init__.py`, the CLI,
CI, pre-commit, and integration into `scripts/training_experiment.py`. Do not
overwrite those files based on this handoff.

## Public configuration API

Available from `rc_car.experiments` and `rc_car.experiments.config`:

```python
load_config(path: str | Path) -> ExperimentConfig
validate_config(config: ExperimentConfig | Mapping[str, object]) -> ExperimentConfig
resolve_config(config: ExperimentConfig | Mapping[str, object]) -> dict[str, Any]
```

`validate_config` returns a frozen validated model. Legacy callers can ignore its
return for validation, but code that consumes effective defaults must use the
resolved result. The root agent reported integrating resolution before simulator
startup and preserving both requested and resolved recipe records in the runner.

`resolve_config` returns an independent JSON-compatible dictionary with recipe
defaults made explicit. Optional absent values are omitted, not emitted as null;
this preserves the historical runner's `dict.get` behavior. It does not read the
environment or initialize tracking. The returned dictionary is a recipe, not a
complete resolved Isaac scene configuration.

Pydantic 2 models use strict validation, `extra="forbid"`, frozen instances,
nonfinite-number rejection, hidden input values in formatted validation errors,
default validation, and instance revalidation. Known limitations are stored as
an immutable tuple and serialize back to a JSON array. No Isaac, Torch, or W&B
imports are needed to load or resolve recipes.

### Preserved inputs and defaults

Both shipped recipes remain accepted:

- `configs/experiments/line_following_15min.json`
- `configs/experiments/multiagent_baseline.json`

Their supplied values survive resolution. Relevant defaults:

- `schema_version=1`, with booleans/floats rejected for the version.
- `enable_obstacles=True`.
- `training_video_frames=90`.
- Missing `max_training_seconds` and `training_video_interval_s` remain omitted.
- Missing `wandb` remains omitted; configuring it defaults its mode to `online`.
- `known_limitations` defaults to an empty sequence.
- PPO defaults match the supported numeric SB3 parameter subset: `n_steps=2048`,
  `batch_size=64`, `n_epochs=10`, `learning_rate=0.0003`, `gamma=0.99`,
  `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.0`, `vf_coef=0.5`,
  `max_grad_norm=0.5`, `use_sde=False`, `sde_sample_freq=-1`.
- Existing CPU tracking-test metadata is explicitly supported through
  `test_only`, `media_provenance`, `note`, and `synthetic_media`.

The schema is deliberately the existing runner's supported numeric subset, not
a representation of every SB3 constructor option. Arbitrary scheduling callables
or additional PPO options require a deliberate schema/API extension.

### Validation rules

- Unknown top-level, PPO, and W&B keys fail rather than being silently ignored.
- Boolean values are not accepted as integer/float configuration values.
- Numeric dimensions, time budgets, video counts, and other positive quantities
  reject zero, negative, nonfinite, and coercible-string values where applicable.
- Explicit null training budgets fail; an absent budget is permitted.
- Seeds are integers in `[0, 2**32 - 1]`, matching NumPy seeding requirements.
- PPO requires more than one rollout sample, batch size greater than one, and a
  batch no larger than `num_envs * n_steps`.
- Partial PPO minibatches are accepted. An initial divisibility restriction was
  removed during review because SB3 supports partial minibatches with a warning.
- Gamma and GAE lambda are bounded to `[0, 1]`; clip range is positive and at
  most one; other coefficient constraints are expressed in the schema.
- SDE sampling frequency is `-1` or a positive integer.
- Duplicate JSON keys are rejected recursively during `load_config`, using
  `object_pairs_hook`. This includes objects inside arrays. Errors identify only
  the repeated key, not its previous/replacement values.

Missing wall-time limits retain historical timestep-only behavior. Even a present
`max_training_seconds` bounds training, not simulator startup, evaluation, or GPU
billing. Configuration validation does not establish correct reward semantics,
calibrated dynamics, or successful driving.

## Tracking API and behavior

```python
start_tracking(config, run_dir, *, job_type="training") -> TrackingRun
log_video(run, path, *, key="training/video", timesteps=0) -> None
```

`TrackingRun` is a protocol for the SDK operations this module uses. W&B is
loaded lazily through `_load_wandb`; importing the package does not initialize an
SDK run or contact a service. Missing optional W&B dependencies produce an
actionable error about installing the tracking dependency group.

`start_tracking`:

1. Validates the recipe and requires a public W&B destination.
2. Honors `WANDB_MODE` over the recipe's mode, accepting only `online` or
   `offline`. The effective mode is recorded in the config supplied to W&B and
   the local identity record.
3. Rejects inherited `WANDB_RUN_ID`, `WANDB_RESUME`, `WANDB_RESUME_FROM`, and
   `WANDB_FORK_FROM` controls. This API starts a fresh run; resume needs a
   separate explicit workflow.
4. Supplies `reinit="create_new"`, a fresh UUID-derived run ID, and
   `resume="never"`. This prevents SDK defaults from returning an already
   active run and mixing experiment identities/configurations.
5. Defines the training metric step relationship and persists public identity
   in `wandb_run.json`: ID, URL, entity, project, and effective mode.
6. Refuses to overwrite an existing identity, including a symlink. Validates SDK
   identity fields before publication.
7. Closes the new run with a failure exit code if metric setup or identity
   persistence fails. A second cleanup exception does not mask the original
   failure.

Credentials are not fields in the recipe or local identity. Tests check that an
environment-injected synthetic credential value is not copied into SDK config
arguments or the identity record. No actual credentials are included here.

### Atomic identity publication

`_persist_identity` serializes before opening a temporary file, writes and
flushes it in the run directory, fsyncs it, then creates a hard link at the final
identity path. The hard-link operation has exclusive-create semantics: a
competing writer causes failure without overwriting its file. The temporary file
is cleaned up afterward. Errors before publication leave no partial identity
that could falsely block a retry.

This design targets the local Mac/Linux filesystems used by the project. It
requires same-filesystem hard-link support. An unusual mounted filesystem that
does not support hard links will fail explicitly and close the run; this was not
tested on a remote object-store/FUSE mount.

`log_video` checks for an existing nonempty `.mp4`, an active run, a nonempty key,
and an integer nonnegative timestep before loading the SDK. It logs under the
requested metric with `training/agent_timesteps`. It does not decode or verify
the MP4 stream; encoding correctness remains the recorder's responsibility.

## Compatibility and integration

The old `scripts/experiment_tracking.py` now re-exports the portable functions.
It adds the repository `src` path so historical standalone script imports work
before every runtime entry point is migrated. New code should import
`rc_car.experiments` directly.

Existing `scripts/check_tracking.py` adds four test metadata fields after its
initial validation. These were inspected and explicitly incorporated into the
schema to keep that flow compatible. The existing training runner, tracking
check, upload script, recipes, and shell orchestrator were inspected before
implementation.

The root agent reported updating the training runner to resolve configuration
before Isaac startup and retain both `requested_config.json` and the effective
`config.json`. Confirm the final integrated file when resuming; it was outside
this subagent's write ownership.

## Independent review findings and fixes

The simulator-review agent independently reviewed this slice. Findings addressed:

1. **W&B can return an already active run by default.** Added explicit new-run
   reinitialization and unique IDs; tests inspect the SDK arguments and repeated
   starts request distinct IDs.
2. **Inherited SDK resume controls can alter run identity.** Reject the four
   environment controls above and explicitly prohibit resuming.
3. **Exclusive open alone could leave a partial identity on failure.** Replaced
   it with serialization-before-write and atomic exclusive publication. Tests
   cover storage flush failure, invalid SDK identity, and a competing writer.
4. **Out-of-range seeds could fail only after GPU startup.** Enforced NumPy's
   unsigned 32-bit seed bounds with accepted-boundary and rejected-boundary tests.
5. **PPO divisibility restriction changed valid historical behavior.** Removed
   the restriction and added an accepted partial-minibatch regression.
6. **Duplicate JSON keys obscured configuration provenance.** Added recursive
   rejection and three regression cases; independently confirmed afterward.

The reviewer reported no remaining blockers after rerunning the final scope.

## Verification already completed

Do not rerun merely to reestablish these results after the pause. Rerun when
changes or environment drift justify it.

Final configuration/tracking verification commands:

```text
.venv/bin/ruff format src/rc_car/experiments tests/experiments
.venv/bin/ruff check src/rc_car/experiments tests/experiments
.venv/bin/ty check src/rc_car/experiments tests/experiments
.venv/bin/pytest -q tests/experiments tests/test_experiment_tracking.py
```

Final result: **80 tests passed, five compatibility subtests passed**. Ruff and
ty passed. The independent simulator reviewer confirmed the final duplicate-key
change and the same `80 passed + 5 subtests` result.

Before the final three duplicate-key tests, full `ty check`, full `ruff check .`,
and `ruff format --check .` passed. At that point Ruff reported 27 checked files.
The root agent owns later full-suite totals and final integration verification.

The tests use fake SDK transport only. A fresh-process import test blocks
Isaac/Torch/W&B imports and network socket connection attempts while loading and
resolving a shipped recipe. The legacy five tests no longer require importing
the actual W&B package to patch its initializer.

## Independent cross-review of the model extraction

Reviewed, without editing:

- `src/rc_car/models/feature_extractor.py`
- `src/rc_car/models/inputs.py`
- `src/rc_car/models/__init__.py`
- `tests/models/test_feature_extractor.py`
- `tests/models/fixtures/legacy_feature_extractor.py`
- Legacy feature-extractor shim under
  `source/car_training/car_training/tasks/manager_based/car_training/agents/`.

Commands completed:

```text
.venv/bin/pytest -q tests/models
.venv/bin/ty check src/rc_car/models tests/models
```

Result: **19 model tests passed** and scoped ty passed. An independent AST
comparison established that the frozen legacy fixture, excluding its added
explanatory docstring, equals the original feature extractor at Git HEAD. The
parity tests compare against the historical implementation, not the new shim
against itself.

The tests check exact initialization/state dictionaries, outputs, input and
parameter gradients in train/eval modes, the historical BatchNorm dummy-forward
update, evaluation-state stability, state-dict round trips, a synthetic CPU PPO
update, and old dotted-class checkpoint loading through the shim.

No blocking findings were identified. Limits remain explicit:

- A normalized `[0, 1]` range assertion cannot distinguish correct binary images
  from an erroneous second division by 255; distribution/semantic checks remain
  necessary.
- Synthetic PPO learning verifies wiring, not driving or simulator behavior.
- The checkpoint namespace fixture avoids Isaac imports. It does not prove
  that the complete real Isaac extension imports correctly after packaging
  changes.
- BatchNorm behavior was preserved intentionally, not endorsed as the best
  policy architecture for future experiments.

## Root tooling and packaging context

The user selected **ty as the primary type checker**. The earlier review's
suggestion of Pyright as an initial gate was superseded; there is no Pyright
gate to implement.

Root work inspected during this slice included:

- A Hatch-built `rc-car-research` package from `src/rc_car`.
- Base NumPy/Pydantic dependencies; an optional `learning` extra with pinned
  Torch/SB3; tracking and development dependency groups.
- Separate root CPU/local and `environments/isaac` runtime lockfiles.
- A Linux CPU-only PyTorch source for root checks, separate from the Isaac
  runtime's CUDA source.
- Ruff scope covering the new portable source/tests and selected owned scripts;
  the frozen historical model fixture is excluded from rewriting.
- ty checking portable source/tests with warnings treated as errors.
- Pytest markers separating GPU and network integration from ordinary checks.
- `scripts/check.py` running Ruff lint, Ruff format check, ty, and pytest excluding
  GPU/network tests.
- Pre-commit invoking the same locked tools through uv.
- CPU GitHub Actions with explicit dependency installation, read-only checkout
  credentials, pinned action revisions, a timeout, and offline W&B environment.

The initial workflow used macOS 14. Cross-review flagged its upcoming runner
retirement, and the root agent confirmed updating it to **macOS 15**. Root owns
the final workflow and lockfile state.

## Remaining limitations and next useful work

- No online W&B calls or real offline-SDK integration runs were performed by
  this subagent. The checks prove our arguments, lifecycle, and persistence logic
  against fakes, not actual service upload or cross-machine offline synchronization.
- No GPU, cloud allocation, renderer, physics, real hardware, or calibrated-car
  tests were performed. Full Isaac integration is still a separate validation gate.
- The package is an initial extraction, not the full eight-month architecture.
  Trace/replay formats, named reward decomposition, observation/history purity,
  real/sim contracts, and deployment interfaces remain future work.
- Ordinary tensor annotations do not establish units, axes, coordinate frames,
  aliasing, or purity. Keep behavioral tests and runtime boundary checks as the
  substantive guarantees.
- Preserve structural changes separately from reward/observation bug fixes,
  simulator upgrades, and scientific changes.
- Before a later remote run, confirm the root runner's resolved-config integration,
  dependency installation, and artifact handoff path. Do not infer that the local
  tests validate cloud setup or billing shutdown.
- A future explicit resume API should record policy/optimizer state and clearly
  distinguish them from simulator/RNG/history/rollout restoration. Do not loosen
  the new-run identity protections to create an implicit resume mechanism.
- Continue using a small tool stack. W&B is the comparison/tracking surface;
  portable local records should remain useful without it. Add a spatial viewer
  or profiler only against a concrete data contract and question.

The user requested the pause before this handoff was written. No further tests,
implementation, cloud actions, or W&B interactions were performed for closeout.
