"""Compatibility import for src.trainer.sft.train_lora."""
import importlib
import sys

_implementation = importlib.import_module("src.trainer.sft.train_lora")

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
