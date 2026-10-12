"""Compatibility import for UI.main."""
import importlib
import sys

_implementation = importlib.import_module("UI.main")
sys.modules[__name__] = _implementation
