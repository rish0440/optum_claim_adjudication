"""Compatibility import for src.orchestration.workflow."""
import importlib
import sys

_implementation = importlib.import_module("src.orchestration.workflow")
sys.modules[__name__] = _implementation
