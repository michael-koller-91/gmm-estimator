"""Tests for example scripts."""

from __future__ import annotations

import subprocess
import sys


def test_gmm_estimator_examples_run() -> None:
    """Test that the example script runs successfully."""
    result = subprocess.run(
        [sys.executable, "examples/gmm_estimator_examples.py"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Running example 1." in result.stdout
    assert "Running example 7." in result.stdout
