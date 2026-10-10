"""Compatibility import for src.orchestration.nodes.invoice_extraction.node."""
import importlib
import sys

_implementation = importlib.import_module("src.orchestration.nodes.invoice_extraction.node")
sys.modules[__name__] = _implementation
