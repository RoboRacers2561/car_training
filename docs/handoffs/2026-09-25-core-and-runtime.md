# Core, tracking review, and runtime handoff

Historical record only: commands and environment layouts below are unsupported.
Use `uv run rc-car` and the [current README](../../README.md). The old command
transcripts describe earlier validation, not the current interface.

Date: 2026-09-25. Branch: `astra-cleanup`.

The user requested a pause to conserve tokens and intends to resume later.
This closeout records completed implementation and review; it does not authorize
resuming implementation or allocating compute during the pause. No cloud or GPU
work was performed by this agent. No implementation or tests were run during
this documentation-only closeout.

## Ownership and scope

This agent implemented the portable numerical slice and its CPU tests, then
independently reviewed the experiment/tracking extraction, runtime packaging,
and documentation. The main agent owns the overall branch, packaging, CLI,
telemetry/lifecycle work and documentation. Other agents own portable models and
experiment configuration/tracking. All changes share the same working tree.

Existing unrelated dirty-worktree changes and prior artifacts belong to the
user. Do not reset, discard, overwrite, or reorganize those as part of resuming.
This workspace is personal; no artifacts were published to team/PL locations.

## Preserved baseline

Before numerical adapter edits, the main agent saved and verified:

- `outputs/refactor-baseline-20260925/source-before-refactor.tar.gz`
- SHA-256: `4e894fb410d9453c5c94834e3fbb50aa3effb4001657863c151b9d352700db73`
- Adjacent `source-provenance.json` records source hashes and dirty state.

This archive preserves source, recipes, docs, tests, and both locks. It is not a
complete backup of every generated asset or experiment output. Existing handoff
archives remain separate. Do not treat the archived source as a proven driving
baseline; it contains inherited defects and fixture-specific behavior.

## Numerical implementation

New portable paths:

- `src/rc_car/core/__init__.py`: lightweight namespace; importing it alone does
  not import Torch or Isaac.
- `src/rc_car/core/_validation.py`: shared metadata/scalar contracts and opt-in
  finite-value checks.
- `src/rc_car/core/geometry.py`: coordinate, heading and velocity transforms.
- `src/rc_car/core/actions.py`: inherited action-to-target mapping.
- `src/rc_car/core/rewards.py`: inherited unweighted reward arithmetic.

Connected Isaac adapters, under
`source/car_training/car_training/tasks/manager_based/car_training/mdp/`:

- `observation.py` calls portable geometry for image-coordinate positions,
  image-frame velocity and headings, and legacy aligned actor velocity.
- `actions.py` calls `legacy_action_targets` in `process_actions`; existing
  joint application and action-buffer operations remain in the adapter.
- `rewards.py` calls portable line-proximity and steering-magnitude kernels;
  state reading, obstacle detection, manager weights and dt remain outside.

The old `mdp/utils/helpers.py::convert_coords` still exists for other legacy
callers. Its formula and the extracted core formula were preserved. Only the
observation adapter's pose conversions were redirected in this slice; broad
helper consolidation was deliberately not performed.

### Exact conventions retained

`convert_coordinates` maps centered world metres to top-left-origin image
pixels with positive image y downward. Dimensions are `(width, height)`.
Coordinates may have arbitrary leading dimensions and at least two trailing
components; trailing z/other components are ignored. With aspect preservation,
the world is letterboxed in the image center. Without it, x/y scale independently.
The inverse is available. The historical convention maps world edges to zero
and image width/height, not to the final pixel center. Output defaults to float32
even for float64 inputs, unless a dtype is explicitly requested.

`world_velocity_to_image` returns `(vx, -vy)` in m/s, not pixels/s, as independent
storage. `heading_to_image_vector` returns `(cos(yaw), -sin(yaw))` for world
heading radians.

`legacy_aligned_velocity` applies `R(-yaw - pi/2)` to planar world velocity. It
optionally divides by a positive normalization speed before rotating, preserving
the historical operation order. The Isaac adapter uses 0.8 m/s when normalized.
For yaw zero, world `(1, 0)` maps approximately to `(0, -1)`. Do not rename this
as a new standard forward/lateral convention or silently change signs.

`legacy_action_targets` accepts an `(N, 2)` floating tensor and computes:

```text
bounded = tanh(raw_actions)
wheel_rad_s = min_wheel_rad_s + 0.5 * (bounded[:, 0] + 1) * (max - min)
steering_rad = max_steering_rad * bounded[:, 1]
```

For inherited driving limits 10..30 rad/s and steering limit 0.6 rad, raw zero
commands 20 rad/s, not stop. For reset limits -15..30 rad/s, raw zero commands
7.5 rad/s. If the trainer already clips raw actions to `[-1, 1]`, effective
driving wheel targets are approximately 12.384058..27.615942 rad/s and steering
only reaches approximately +/-0.456956 rad. The tanh and this reduced effective
range were preserved. The adapter still sends equal targets to both rear wheels
and equal steering angles to both front joints; the extraction does not add
Ackermann wheel-angle conversion.

`legacy_line_proximity` returns `1 - distance_px / distance_scale_px`, or zero
when the supplied obstacle-present mask is true. Default distance scale is 40
pixels. Distances beyond the scale give negative reward. This is legacy contour
distance, not validated centerline or signed route progress.

`legacy_steering_penalty` returns `-abs(steering_rad) / max_steering_rad`, or zero
when the obstacle gate is true. It penalizes magnitude, not steering rate or jerk.
The kernels neither apply manager weights nor multiply by dt. Existing obstacle
image reads and their inherited history side effects were not changed.

### Contracts and deliberate behavior change

Shape, floating-dtype and scalar configuration checks are cheap and run directly.
Finite tensor-value checks are opt-in via `validate_values`; they synchronize a
CUDA tensor with the host and should be used in preflight/debugging rather than
silently added to every GPU hot-path call. Local contracts are not evidence that
the whole simulator satisfies them.

One intentional scientific/behavior change was made: old `robot_vel` sliced
`root_com_lin_vel_w[:, :2]` and negated y in place, mutating the simulator's
cached velocity tensor. Repeated reward/observation reads could flip that source
state again and change subsequent actor input. It now returns a new tensor
without touching the source. First-read mathematical values are retained, but
whole trajectories need not match the old defective system. Main records this
as `astra-foundation-v1` and explicitly disclaims trajectory parity.

No other objective, normalization, action-range, camera timing, history sampling,
reset or termination redesign was made by this numerical slice.

## Numerical test and review evidence

Files:

- `tests/core/test_geometry.py`
- `tests/core/test_actions.py`
- `tests/core/test_rewards.py`
- `tests/core/test_isaac_math_adapters.py`

Last scoped execution: **29 tests passed**, including Hypothesis cases.
`ruff check`, `ruff format --check` on the owned core/tests, and `ty check
src/rc_car/core` all passed. Tests cover coordinate landmarks/letterboxing and
roundtrips, speed invariance under rotation, immutability, empty batches, shape
and finite-value contracts, action golden values/limits/autograd, reward signs
and gating, and repeated-read invariance through the actual adapter body.

Adapter tests extract selected function bodies and their real `rc_car` imports
from source AST, supplying lightweight fake state readers. This provides a
meaningful CPU check of adapter wiring and the velocity mutation fix. It does
**not** import the full Isaac module, start a simulator, validate manager
scheduling, or prove GPU integration. That limitation is explicit in the test.

The model agent independently reviewed the numerical core and adapter diffs,
reran the 29 tests, and reported no numerical regression. Its snapshot comparison
found bit-exact retained coordinate behavior. A documentation clarification about
tanh versus trainer-side clipping was incorporated; the effective +/-0.456956
rad behavior is documented and tested.

## Independent experiment/tracking review

Reviewed `src/rc_car/experiments/{config,tracking}.py`, their tests,
`scripts/experiment_tracking.py` compatibility imports, and the legacy tests.

Initial review found concrete issues, all subsequently fixed by the author:

1. W&B's default initialization can return an already active run. Installed
   W&B source confirmed this. Fresh starts now request `reinit="create_new"`,
   an explicit random ID, and `resume="never"`; inherited run-ID/resume/fork
   environment controls are rejected for this fresh-run API.
2. Opening the final identity before serialization/write could leave an empty
   or truncated file and block retries. The author now validates/serializes,
   writes and fsyncs a temporary file, publishes it through an exclusive atomic
   hard link, and cleans the temporary path. Race and failure tests cover no
   clobber and no partial final identity. Supported deployment filesystems are
   local Mac/Linux filesystems; unusual filesystems may not support hard links.
3. Seeds at or above `2**32` passed configuration validation but SB3 calls NumPy's
   narrower seed API. Train/eval seeds are now constrained to `[0, 2**32)`.
4. The first schema rejected non-divisible PPO minibatches even though legacy
   validation and installed SB3 support them. Partial minibatches are again
   accepted without changing their values.
5. JSON duplicate keys were initially accepted with last-write-wins behavior.
   `load_config` now rejects duplicate keys recursively, including objects
   inside arrays; error strings do not echo either value.

The compatibility runner initially ignored newly resolved defaults, allowing a
schema-defaulted missing `ppo` to fail later at `config["ppo"]`. Main fixed it
to use `resolve_config(load_config(...))`. Requested and resolved configurations
are now distinct records.

Final independent scoped result: **80 tests passed plus five compatibility
subtests**. The previous 77-test scope also passed Ruff and ty independently;
the author reported Ruff/ty passing after the final three duplicate-key tests.
Import-isolation tests block Torch/Isaac/W&B imports and socket connection and
pass. SDK tests are fake/mock transport tests; no new live W&B upload or offline
SDK end-to-end integration was performed. No secrets were included in review
output or this handoff.

## Runtime packaging verification

Reviewed root `pyproject.toml`/`uv.lock`,
`environments/isaac/pyproject.toml`/`uv.lock`,
`source/car_training/setup.py`, and the pinned setup script.

- Root package is `rc-car-research`, installed from `src/rc_car` with base
  NumPy/Pydantic dependencies and optional `learning` Torch/SB3 dependencies.
- Root Linux learning resolution selects the CPU Torch index; Mac uses its
  PyPI wheel. Root lock contains no `nvidia-*` or Isaac packages.
- Isaac explicitly includes the editable root portable package, **without**
  requesting its learning extra, and supplies its own CUDA dependencies.
- The legacy `car-training` package declares the portable dependency too.
- Root lock: `uv lock --check --offline` passed, resolving 66 packages.
- Isaac lock: `uv lock --project environments/isaac --check --offline` passed,
  resolving 226 packages. This used the existing cache in an auto-approved
  metadata-only escalation after the sandbox denied cache initialization; no
  install, network or compute allocation occurred.
- Parsed comparison against the source snapshot found **all 225 pre-existing
  Isaac package versions and package sources unchanged**. The sole additional
  package is editable `rc-car-research==0.1.0`.

Retained pins include Isaac Sim `5.1.0.0`, Torch `2.7.0+cu128`, torchvision
`0.22.0+cu128`, torchaudio `2.7.0+cu128`, and SB3 `2.7.0`. Isaac Lab remains the
editable v2.3.0 checkout guarded by the unchanged setup-script commit check
(`3c6e67bb5c7ada942a6d1884ab69338f57596f77`). Package component versions in that
checkout are not the overall Lab release number; do not misinterpret them.

An empty temporary uv cache sufficed for the root check but could not verify
Isaac metadata offline because cached build requirements were absent. That was
a cache availability issue, not a lock mismatch. Do not perform an unnecessary
runtime upgrade in response. Isaac 5.1 is retained as a historical environment;
a supported-runtime migration is a separately tested future change.

## Documentation review

Reviewed `README.md`, `docs/architecture.md`,
`docs/decisions/0001-portable-foundation.md`, and the progress update in
`docs/refactor.md`. Main archived the old README in `docs/history/`.

The docs correctly distinguish the first portable stage from completion of the
eight-month platform and explicitly state that the refactored integration has
not run on GPU. Main corrected the README GPU command to **fixture-only**:
the current runner unconditionally calls `configure_fixture` and requires
`fixture.json`; arbitrary recovered original USD bundles are not supported by
that command. Main also reports applying the two final wording fixes: label the
old findings table as pre-refactor evidence and use `rc_car.core` for the
portable import example.

CPU CI is configured for Linux and macOS with pinned action SHAs and read-only
permissions. A local check is not evidence that remote CI has executed.

## Remaining scientific and integration work

Do not infer completion from the large CPU suite. The following are still open:

- Full Isaac imports, app startup, rendering, action/observation shapes, movement,
  PPO update and cleanup must be checked on a bounded GPU smoke test.
- Observation history still mutates through observation/reward reads. A correct
  history abstraction requires one explicit update per control step and selective
  reset in the actual environment lifecycle; no disconnected utility was added.
- Ego-car masking, image metadata/normalization, basic reward argument/helper
  leakage, reset state and termination/truncation behavior remain inherited.
- Independent parallel worlds versus interacting agents need an explicit
  contract; `num_envs` currently serves ambiguous roles in legacy code.
- Full synchronized state/observation/action/reward traces and a calibrated
  planar simulator are planned, not implemented.
- The current generated fixture uses uncalibrated geometry, mass, friction and
  actuation. Its twelve circular tracks and 30 Hz control override do not
  reproduce the historical twelve-track/10 Hz experiment or real micro-car
  dynamics. There is no reproduced 15-minute convergence claim.
- A conventional PID/pure-pursuit baseline, held-out hardware trajectories,
  actuator calibration, latency/dropout tests and meaningful driving evaluation
  are needed before simulator or learner superiority can be claimed.
- Sim physics throughput does not establish the real 20+ car network capacity.
- SB3 remains a historical learner; skrl/MAPPO or RLlib comparison should follow
  a scoped research question, not a wholesale speculative backend rewrite.
- Exact continuation requires simulator, RNG, history and partial-rollout state;
  passing checkpoint/model reload tests do not establish that capability.

## Safe resumption sequence

Wait for the user's request to resume. Then read this handoff together with the
main-agent handoff, current `docs/architecture.md` and `docs/refactor.md`. Confirm
working tree and branch with read-only commands before edits:

```bash
git branch --show-current
git status --short
```

If implementation changed or checks are needed after resumption, the existing
environment can run the single local gate without concurrent environment sync:

```bash
.venv/bin/python scripts/check.py
```

For a deliberately recreated development environment, use the full optional
dependency selection consistently so uv does not remove packages between runs:

```bash
uv sync --locked --extra learning --group tracking
uv run --locked --extra learning --group tracking python scripts/check.py
uv run --locked --extra learning --group tracking rc-car show-config configs/experiments/multiagent_baseline.json
```

These are local setup/check commands, not instructions to start a paid instance.
Do not run concurrent uv synchronization across agents. Avoid rerunning broad
checks simply to reproduce already-recorded evidence unless code/environment
has changed or a specific concern requires it.

Next substantive work should reconcile the main agent's remaining tasks and
apply small, explicit behavioral changes with regression tests. Before any GPU
work, define a bounded smoke gate, artifact/video destination, source/asset/run
identity and provider lifetime cleanup. A detached job or process timeout does
not stop GPU billing. Never allocate or resume compute during this pause.
