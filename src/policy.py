"""Compatibility import for src.adjudication.policy."""
import importlib
import sys

_implementation = importlib.import_module("src.adjudication.policy")
sys.modules[__name__] = _implementation
