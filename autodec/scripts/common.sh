#!/usr/bin/env bash

ensure_fast_sampler() {
    if python - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("superdec.fast_sampler._sampler") is None:
    sys.exit(1)
PY
    then
        return 0
    fi

    echo "--- building superdec.fast_sampler._sampler"
    python setup_sampler.py build_ext --inplace
}
