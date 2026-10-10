"""Compatibility import for src.trainer.sft.evaluate."""
import importlib
import sys

_implementation = importlib.import_module("src.trainer.sft.evaluate")

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
