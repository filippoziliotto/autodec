#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
source "${SCRIPT_DIR}/common.sh"
ensure_fast_sampler
python -m autodec.eval.run --config-name eval_test "$@"

