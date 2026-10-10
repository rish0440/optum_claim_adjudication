"""Compatibility import for src.orchestration.state."""
import importlib
import sys

_implementation = importlib.import_module("src.orchestration.state")
sys.modules[__name__] = _implementation
