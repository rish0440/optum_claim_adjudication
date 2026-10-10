"""Compatibility import for src.inference.local."""
import importlib
import sys

_implementation = importlib.import_module("src.inference.local")

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
