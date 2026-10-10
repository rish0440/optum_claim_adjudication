"""Compatibility import for src.orchestration.nodes.csv_validation.node."""
import importlib
import sys

_implementation = importlib.import_module("src.orchestration.nodes.csv_validation.node")
sys.modules[__name__] = _implementation
