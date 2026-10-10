"""Compatibility import for src.orchestration.nodes.adjudication.node."""
import importlib
import sys

_implementation = importlib.import_module("src.orchestration.nodes.adjudication.node")
sys.modules[__name__] = _implementation
