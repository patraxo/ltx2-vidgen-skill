#!/usr/bin/env bash
# Deploy ltx2-fast to Modal.
#
# No secrets required: the app uses no auth (open endpoints) and no HuggingFace
# key. Model weights are provisioned to the Modal volume out of band (see the
# README "Weights" section); the build pulls only public components.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if ! command -v modal >/dev/null 2>&1; then
  echo "❌ modal CLI not found.  pip install modal && modal token new"
  exit 1
fi

echo "🚀 deploying deploy/ltx2_model.py ..."
# PYTHONPATH=repo root so the utils/ package resolves for add_local_python_source.
PYTHONPATH="${REPO_ROOT}" modal deploy deploy/ltx2_model.py

echo "✅ deployed."
echo "   smoke test:     PYTHONPATH=. modal run tests/smoke_test.py"
echo "   real-image i2v: PYTHONPATH=. modal run deploy/ltx2_model.py::smoke_real --image-path /path/to.jpg"
