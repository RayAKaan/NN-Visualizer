"""Shared pytest fixtures / config for NN-Visualizer backend tests."""
import os
import sys

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import pytest

# Heavy tests (real inference through the registry / prediction service) are
# skipped unless the caller opts in via NN_RUN_HEAVY=1.  They require network
# access for the first weight download and a TensorFlow-capable environment.
RUN_HEAVY = os.environ.get("NN_RUN_HEAVY", "") == "1"

requires_weights = pytest.mark.skipif(
    not RUN_HEAVY,
    reason="set NN_RUN_HEAVY=1 to run real model inference tests",
)

# VGG16 needs far more RAM than CI sandboxes usually allow; run it only when
# explicitly requested via NN_RUN_VGG16=1 + NN_MIN_MEMORY_MB>=3500.
requires_vgg16 = pytest.mark.skipif(
    os.environ.get("NN_RUN_VGG16", "") != "1"
    or int(os.environ.get("NN_MIN_MEMORY_MB", "0") or "0") < 3500,
    reason="set NN_RUN_VGG16=1 and NN_MIN_MEMORY_MB>=3500 to test VGG16",
)


@pytest.fixture(scope="session")
def backend_root():
    return BACKEND
