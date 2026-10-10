"""Compatibility import for src.orchestration.graph."""
import importlib
import sys

_implementation = importlib.import_module("src.orchestration.graph")
sys.modules[__name__] = _implementation
