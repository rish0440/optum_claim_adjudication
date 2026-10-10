"""Compatibility import/CLI for the synthetic annotation generator."""
import importlib
import sys

_implementation = importlib.import_module("src.trainer.sft.annotation.generator")

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
