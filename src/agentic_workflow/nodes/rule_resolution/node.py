"""Compatibility import for src.orchestration.nodes.rule_resolution.node."""
import importlib
import sys

_implementation = importlib.import_module("src.orchestration.nodes.rule_resolution.node")
sys.modules[__name__] = _implementation
