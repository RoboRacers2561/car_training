# September 26: local-only operations and staged research handoff

This supersedes the cloud-scope section of `2026-09-26-resume.md`, not its
historical evidence. Branch `astra-cleanup`; base commit
`b407792bcda16dbc1bb39db2d89e821ce8578b6d`. Changes are still uncommitted in the
existing dirty worktree. No reset, stash, commit, push or team publication.

## Authority and cloud closure

The user briefly approved one bounded session, then explicitly requested immediate
deletion and local-only work because they were leaving. That latest restriction
controls. Do not treat the earlier approval as permission to restart anything.
The research/demonstration/sweep requests were subsequently clarified as mostly
post-refactor; root chose the staged [work plan](../work-plan.md).

- Instance `astra-validate-0926`, ID `eakj5wmuh`, type
  `massedcompute_A6000_plus`, quote $0.68/hour (historical quote, not live price).
- Allocation started 2026-09-27 03:32:31 UTC; deletion completed and
  `brev ls --json` returned `{"workspaces": null}` at 03:37:44 UTC.
- Only host probing: RTX A6000, driver 580.126.09, Ubuntu 22.04, roughly 226 GB
  free; no source upload, installs, training or remote experiment artifacts.
- Local originals and prior backups were preserved. Closure receipt:
  `outputs/cloud-validation/20260926/session.json`. No invoice audit performed.
- Root used the Brev CLI skill for lifecycle checks/deletion. No worker allocated
  anything. No new cloud calls are needed to continue the local refactor.

No independent control-plane expiry has been configured/tested. A timeout bounds
the normal job process group, not billing. A future new go must be attended or
include separately tested independent expiry; retain one shared environment.

## New implementation

- `src/rc_car/ops/bundles.py`: explicit regular-file allowlists, strict hashed
  manifests, archive/member verification, safe no-overwrite extraction. Checksums
  are consistency, not sender authentication. No network or cloud operations.
- `ops/prepare_session.py`: recipe validation, pre/post source captures, full local
  gate, source/fixture bundle verification and bundle-vs-capture hash equality.
  Rejects edits during checks/packing. Fresh output only.
- `scripts/capture_provenance.py`: explicit source selection including ops and
  experiment protocols; fail-closed enumeration/stat, reject symlinks/known
  credential-like files, record absent optional roots; no silent unreadable-tree
  omissions. Text-source allowlist, not a whole checkout backup or secret scanner.
  Its legacy `--archive` option still assumes no concurrent source edits and does
  not rehash tar members after packing; use the verified prepared-bundle workflow
  for the new session capsule.
- `ops/setup_runtime.sh` and `run_gpu_checks.sh`: explicit Linux/EULA gates,
  repository lock, setup/job deadlines, signal cleanup, offline W&B isolation,
  logs and exit receipt. Four modes cover Basic enabled/disabled obstacles and
  shared Multi. No instance creation/deletion or billing enforcement.
- `scripts/run_experiment.sh`: externally supervised mode avoids nested timeout
  process groups escaping the outer deadline. Standalone phase bounds remain.
- `scripts/smoke_test.py`: actual fixed-action camera video, reset-excluded
  per-car movement, initial/final weights and PPO epoch counter. Fixed-action
  smoke video is not trained-policy driving evidence. GPU execution pending.
- `ops/inspect_results.py` / `evidence.py`: finite CPU tensor comparison using
  restricted weights-only loading, changed parameters rather than BatchNorm-only
  buffers, matched initial states/seeds, each-car movement, decoded video frames.
  Never claims driving success. Archive failed evidence anyway.
- `ops/bundle.py`: fixture exact-coverage verification and one-job result backup;
  skipped symlinks disclosed. Cross-machine W&B sync remains unverified.
- `ops/brev_commands.py`: print-only future attended checklist; no execution API.
- `docs/work-plan.md`, `tickets.md`, `reproduction-contract.md`, `experiments/`
  organize prerequisites, historical differences, hypotheses and evidence formats.
  No real-data experiment or W&B Sweep was launched in this local phase.

## Review and validation

Three agents performed scoped work and independent cross-reviews; root integrated
and wrote the plan/runbook. Reviews caught and fixed nested-timeout group escape,
exit-status write masking, source drift during preparation, silent scan errors,
cleanup masking and report/log-preservation ambiguities.

The combined local gate passed **366 tests** with Ruff lint/format and ty clean.
Both locks validate offline (66 root / 226 Isaac packages); the existing local
environment's 61 packages pass compatibility checking. Shell syntax/diff checks
pass. Mac `--plan` commands perform no setup/GPU/cloud actions. These are local
results, not a remote CI run, real Linux/Isaac validation or physics/calibration
proof. The shell suite uses a fake host; one regression runs the actual rehearsal
wrapper around fake Isaac processes to check timeout propagation.

The historical `baseline-002` inspector run passed its 11 evidence checks, saved at
`outputs/cloud-validation/20260926/historical-baseline-inspection.json`. It tests
the new evidence reader against existing artifacts, not a new policy or new
comparison analysis. Both videos decoded to the expected 225 frames and actual
policy tensors changed. Its `driving_proven` remains false.

## Frozen local handoff

Final preparation location:
`outputs/gpu-sessions/astra-ready-20260926/`.
**Require its `session.json` and a successful `SHA256SUMS` verification before
using it.** The preparation process writes the definitive source identity,
local-check log and verified bundles only after its gates. A failed/incomplete
directory is not a ready session; never infer readiness from its name.

```sh
(cd outputs/gpu-sessions/astra-ready-20260926 && shasum -a 256 -c SHA256SUMS)
.venv/bin/python ops/brev_commands.py --instance astra-next \
  --session outputs/gpu-sessions/astra-ready-20260926
```

The second command only prints a checklist. Detailed local unpacking, future
host staging, setup, serial gates, artifact inspection/backup and deletion are
in [ops/README](../../ops/README.md). Changes after freezing require another fresh
preparation. Do not modify the capsule and retain its old source identity.

The earlier source archive remains unchanged at
`outputs/refactor-validation-20260926/source-after-local-validation.tar.gz`, SHA256
`ad048f6c111a07e681dba6525a76afd491fd298e98d69b6321482ee8147d7595`.
It predates these ops changes. September 25 baseline archives also remain intact.

## Next local work (no questions or GPU required)

1. T03: isolate history/ego/image-range defects with characterization tests; apply
   one audited correction per behavior version, preserving the old reference.
2. T04: trace reset/time/termination contracts and agent-versus-world ownership.
3. T06: recover final benchmark identity. Fifteen older original USDs exist;
   validated final track/model/calibration and a numerical “converged” criterion
   do not. The report's 15-minute single-car result cannot be equated with the
   retained four-car fixture or a passing CPU gate.
4. T11: consolidate supported entry points only after identifying their callers.
5. Keep T05 GPU acceptance blocked on new go; demonstrations, multi-seed
   reproduction and hyperparameter experiments follow the foundation gates.

Security: rotate the W&B key previously pasted into chat before future online
use. Never copy it into recipes, scripts, artifacts or restart notes. The offline
SDK persistence gate passed earlier; portable offline sync/online artifact
permissions and native Sweep identity integration are distinct unfinished work.
