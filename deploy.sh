#!/usr/bin/env bash
# Deploy ltx2-fast-inference to Modal.
#
# Uses uv for environment management (deps live in pyproject.toml). Model weights
# are provisioned to the Modal volume on first build (public components only).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "❌ uv not found. Install: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

echo "📦 syncing environment (uv) ..."
uv sync --quiet

if ! uv run modal token validate >/dev/null 2>&1; then
  echo "🔑 Modal not authenticated — run:  uv run modal token new"
  exit 1
fi

echo "🚀 deploying deploy/ltx2_model.py ..."
# PYTHONPATH=repo root so the utils/ package resolves for add_local_python_source.
PYTHONPATH="$REPO_ROOT" uv run modal deploy deploy/ltx2_model.py

echo "✅ deployed."
echo "   smoke:          PYTHONPATH=. uv run modal run tests/smoke_test.py"
echo "   real-image i2v: PYTHONPATH=. uv run modal run deploy/ltx2_model.py::smoke_real --image-path /path/to.jpg"
echo "   full verify:    PYTHONPATH=. uv run modal run tests/ship_verify.py"
