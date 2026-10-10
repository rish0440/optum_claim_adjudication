"""Compatibility import for src.trainer.sft.dataset."""
import importlib
import sys

_implementation = importlib.import_module("src.trainer.sft.dataset")

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
