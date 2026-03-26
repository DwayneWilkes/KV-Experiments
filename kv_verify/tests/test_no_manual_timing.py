"""Tests that experiment functions do not contain redundant manual timing code.

The ExperimentTracker's stage() context manager and @stage decorator
handle timing automatically. Manual t0 = time.monotonic() / elapsed
patterns are redundant boilerplate that should not exist in experiment
run functions.
"""

import ast
import inspect
import textwrap

from kv_verify.experiments.f01_falsification import run_f01a, run_f01b, run_f01c
from kv_verify.experiments.f02_held_out_input_control import run_f02
from kv_verify.experiments.f03_cross_model_input_control import run_f03
from kv_verify.experiments.f04_cross_condition_validity import run_f04


def _has_manual_timing(func) -> list[str]:
    """Check if a function contains manual timing patterns.

    Returns a list of violations found (empty if clean).
    Looks for:
      - time.monotonic() calls
      - tracker.log_metric("elapsed_seconds", ...) calls
      - "elapsed_seconds" keys in dict literals
    """
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)

    violations = []

    for node in ast.walk(tree):
        # Check for time.monotonic() calls
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "monotonic"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "time"):
            violations.append(
                f"time.monotonic() call at line {node.lineno}"
            )

        # Check for tracker.log_metric("elapsed_seconds", ...)
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "log_metric"
                and len(node.args) >= 1
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "elapsed_seconds"):
            violations.append(
                f'tracker.log_metric("elapsed_seconds", ...) at line {node.lineno}'
            )

        # Check for "elapsed_seconds" as a dict key
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if (isinstance(key, ast.Constant)
                        and key.value == "elapsed_seconds"):
                    violations.append(
                        f'"elapsed_seconds" dict key at line {key.lineno}'
                    )

    return violations


class TestNoManualTiming:
    """Verify that experiment run functions have no redundant manual timing."""

    def test_f01a_no_manual_timing(self):
        violations = _has_manual_timing(run_f01a)
        assert violations == [], (
            f"run_f01a contains manual timing code (tracker handles this):\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_f01b_no_manual_timing(self):
        violations = _has_manual_timing(run_f01b)
        assert violations == [], (
            f"run_f01b contains manual timing code (tracker handles this):\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_f01c_no_manual_timing(self):
        violations = _has_manual_timing(run_f01c)
        assert violations == [], (
            f"run_f01c contains manual timing code (tracker handles this):\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_f02_no_manual_timing(self):
        violations = _has_manual_timing(run_f02)
        assert violations == [], (
            f"run_f02 contains manual timing code (tracker handles this):\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_f03_no_manual_timing(self):
        violations = _has_manual_timing(run_f03)
        assert violations == [], (
            f"run_f03 contains manual timing code (tracker handles this):\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_f04_no_manual_timing(self):
        violations = _has_manual_timing(run_f04)
        assert violations == [], (
            f"run_f04 contains manual timing code (tracker handles this):\n"
            + "\n".join(f"  - {v}" for v in violations)
        )
