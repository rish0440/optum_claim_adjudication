"""Compatibility import for app.main."""
import importlib
import sys

_implementation = importlib.import_module("app.main")
sys.modules[__name__] = _implementation
