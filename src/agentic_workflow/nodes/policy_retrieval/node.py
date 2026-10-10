"""Compatibility import for src.orchestration.nodes.policy_retrieval.node."""
import importlib
import sys

_implementation = importlib.import_module("src.orchestration.nodes.policy_retrieval.node")
sys.modules[__name__] = _implementation
