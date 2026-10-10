"""Compatibility import for src.config.settings."""
import importlib
import sys

_implementation = importlib.import_module("src.config.settings")
sys.modules[__name__] = _implementation
