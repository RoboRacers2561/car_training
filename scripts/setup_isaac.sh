#!/usr/bin/env bash
# Upstream Isaac Lab needs an editable checkout: wheels omit config/apps assets.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
ISAACLAB_REV=3c6e67bb5c7ada942a6d1884ab69338f57596f77 # v2.3.0
if [[ ! -e .isaaclab ]]; then
    git clone --branch v2.3.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git .isaaclab
fi
if [[ "$(git -C .isaaclab rev-parse HEAD)" != "$ISAACLAB_REV" ]]; then
    echo 'Unexpected Isaac Lab commit. Move the existing .isaaclab checkout aside and rerun.' >&2
    exit 1
fi
if [[ -n "$(git -C .isaaclab status --porcelain)" ]]; then
    echo 'Isaac Lab checkout has local changes; refusing a non-reproducible install.' >&2
    exit 1
fi
uv sync --locked --project environments/isaac
uv pip check --python environments/isaac/.venv/bin/python
