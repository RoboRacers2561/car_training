# Restart after the September 27 GPU validation

Follow-up: the three offline runs have now been uploaded and independently
server-verified; see `outputs/wandb-upload-20260927/summary.json` and the current
README. The text below records the original GPU-session closure. The working
tree now also has a single `config.yaml`/no-argument workflow, new W&B recovery
tools and cleanup. Its final local gate passed 681 tests, Ruff and ty; those changes
need a new source capture before future GPU use. Debug images/action logs were
archived under `outputs/archive/pre-refactor-debug` with verified checksums, and
obsolete Flake8/supplied-media tracking/alternate job entry points were retired
with recoverable copies. Active runtime and checkpoint compatibility code remains.

Branch: `astra-cleanup`, dirty and uncommitted. Preserve all work. No GPU remains:
`astra-smoke-0927` / `xwzk0f40e` was confirmed absent at 21:07:53 UTC after verified
backup. The 53 min 44 s allocation corresponds to about $0.61 at the quoted $0.68/hour,
not an invoice. The ten-hour permission was a ceiling, not a persistent allocation.

## Completed, with evidence

- [E001](../../experiments/E001-foundation-integration/README.md): four frozen
  foundation integration gates passed, independently inspected.
- [E006](../../experiments/E006-900s-fixture-baseline/README.md): 900.051 s training,
  206,376 collected / 196,608 inferred optimized car samples, 160 PPO epochs. Matched
  15-second evaluation: off-track resets 66 → 0, aggregate travel 98.773 → 197.148 m. No full
  lap, original-team convergence or held-out generalization claim.
- [Actual comparison video](../../outputs/cloud-validation/20260927-run02/remote/outputs/baselines/basic-900s-01/experiment/start-vs-end.mp4).
- [Session receipt](../../outputs/cloud-validation/20260927-run02/session.json):
  240 files / 200,059,483 bytes verified against remote hashes and independently
  audited for exact coverage. All three offline W&B streams and artifact staging,
  16 intermediate baseline checkpoints, 8 training clips and 22 diagnostic logs saved.
  Runtime binaries were intentionally not copied; rebuild from the pinned setup.

Reference executable source remains the immutable 153-file capsule
`outputs/gpu-sessions/astra-ready-20260926`, not current branch HEAD alone.
The current working code includes a new observation contract and cannot silently
load those legacy checkpoints. To rerun E006, use its original frozen source.

## Next bounded step: E003 image contract

[E003 protocol/status](../../experiments/E003-policy-image-encoding/README.md) and
prepared 168-file capsule `outputs/gpu-sessions/astra-images-v3-20260927` are ready.
453 CPU tests, Ruff/format/ty and independent reviews passed. Fixture entries are
byte-identical to the reference. No candidate GPU execution occurred.

Automatic tool review blocked the changed private payload because earlier
explicit upload approval covered 153 files only. **Get explicit approval for the
168-file snapshot and the new destination before transfer.** No candidate file
reached the deleted host. Do not reattempt via an indirect transfer or narrower
delta as a workaround. Check current Brev inventory/pricing and a fresh attended
budget before allocating one environment; root alone controls lifecycle.

Verify/extract the prepared capsule using `ops/bundle.py`; preserve its captured
identity even though these later reports were updated after capture. Follow
`ops/README.md` for setup, then run the four E003 gates serially in fresh paths,
with offline tracking and the preregistered old/new motion/reward comparisons.
Do not start a sweep or another long run merely because local tests passed.

## W&B and remaining work

W&B recording is offline, not uploaded. `scripts/prepare_wandb_handoff.py` checks
backed-up bytes and builds a later upload plan; it does not sync. Plans for Basic
rehearsal and E006 are under the session directory. SDK artifact references can
retain absolute GPU paths; validate a bounded synthetic online sync/re-upload
flow and verify server manifests before claiming success. Rotate the previously
exposed API credential; do not paste it into chats or transfer home credentials.

The durable E006 analyzer/tests recompute raw trajectory metrics and correctly
exclude reset jumps and the unoptimized final partial rollout. The current
backlog is [tickets](../tickets.md): image-contract GPU proof first, then separate
ego/history/time alignment, held-out/endurance metrics and original-source
recovery. Framework/platform migration and sweeps are not prerequisites.
