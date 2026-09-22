#!/usr/bin/env bash
# Always launch from the repo root; do not mix the Mac/dev and Linux/Isaac envs.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
exec uv run --locked --project environments/isaac python "$@"
