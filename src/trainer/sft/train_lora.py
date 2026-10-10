"""Compatibility entry point for the canonical manual QLoRA trainer."""
import importlib
import sys

_implementation = importlib.import_module("src.trainer.sft.train.train_qlora")
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
