"""Compatibility import for src.inference.remote_client."""
import importlib
import sys

_implementation = importlib.import_module("src.inference.remote_client")
sys.modules[__name__] = _implementation
