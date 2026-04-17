#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

echo "--- running AutoDec phase 1"
bash "${SCRIPT_DIR}/run_phase1.sh" "$@"

echo "--- running AutoDec phase 2"
bash "${SCRIPT_DIR}/run_phase2.sh" "$@"

echo "--- AutoDec phase 1 and phase 2 finished"
