"""Compatibility import for src.orchestration.nodes.response_builder.node."""
import importlib
import sys

_implementation = importlib.import_module("src.orchestration.nodes.response_builder.node")
sys.modules[__name__] = _implementation
