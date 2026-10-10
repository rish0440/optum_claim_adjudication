"""Compatibility import for src.trainer.sft.training_data."""
import importlib
import sys

_implementation = importlib.import_module("src.trainer.sft.training_data")
sys.modules[__name__] = _implementation
