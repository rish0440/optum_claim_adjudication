"""Manual vLLM launch preparation; fail closed on adapter/dependency mismatches."""
import argparse
import importlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import sys

from packaging.requirements import Requirement
from packaging.version import Version

from src.config.models import sft_adapter_pointer
from src.inference.vlm_client import ADAPTER_SERVED_MODEL, BASE_MODEL_ID, BASE_SERVED_MODEL
from src.utils.paths import ROOT

REFERENCE_VLLM_VERSION = "0.11.2"
SERVING_STACK = {"vllm": "0.11.2", "torch": "2.9.0", "torchvision": "0.24.0",
                 "transformers": "4.57.6"}
SERVING_PACKAGES = {**SERVING_STACK, "httpx": None, "Pillow": None}


def resolve_adapter(pointer_path=None, model_id=BASE_MODEL_ID, max_rank=8):
    """Reuse Phase 5's checksum/weight/run verification without altering weights."""
    pointer_path = Path(pointer_path or sft_adapter_pointer()).resolve()
    if not pointer_path.is_file():
        raise ValueError(f"Missing successful adapter pointer: {pointer_path}")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("model_id") != model_id:
        raise ValueError("Adapter pointer base-model ID differs from the selected base model")
    # Lazy import: basic HTTP inference never imports Torch or trainer modules.
    from src.trainer.sft.train.checkpoints import latest_adapter
    adapter = latest_adapter(pointer_path.parent)
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != model_id or config.get("r") != 8 or max_rank < 8:
        raise ValueError("Phase 6 expects this base model and rank-8 adapter; max LoRA rank must be >=8")
    forbidden = ("use_dora", "use_rslora", "use_qalora", "modules_to_save", "rank_pattern", "alpha_pattern", "lora_bias")
    if any(config.get(name) for name in forbidden):
        raise ValueError("Unsupported adapter variant; conventional language-attention LoRA is required")
    template = adapter / "chat_template.jinja"
    if not template.is_file():
        raise ValueError("The saved Qwen chat_template.jinja is missing")
    training = json.loads((adapter.parent / "run_config.json").read_text(encoding="utf-8"))
    return {"path": adapter, "model_id": model_id, "rank": config["r"], "template": template,
            "min_pixels": training["min_pixels"], "max_pixels": training["max_pixels"]}


def check_environment():
    """Check the isolated serving stack and basic CUDA access, not serving kernels."""
    installed, errors = {}, []
    active_prefix = Path(sys.prefix).resolve()
    active_executable = Path(sys.executable).absolute()
    training_envs = {(parent / ".venv").resolve() for parent in (ROOT, *ROOT.parents)}
    if active_prefix == Path(sys.base_prefix).resolve():
        errors.append(f"The serving interpreter is not running in an isolated virtual environment: {active_prefix}")
    elif active_prefix.name == ".venv" or active_prefix in training_envs:
        errors.append(f"The serving environment must be separate from the QLoRA training .venv: {active_prefix}")
    elif active_executable.parent.resolve() != (active_prefix / "bin").resolve():
        errors.append(f"Interpreter executable {active_executable} does not belong to active environment {active_prefix}")
    if errors:
        executable = Path(sys.executable).parent / "vllm"
        return {"versions": installed, "cuda": {"available": False, "device": None,
                "verification": "not checked until an isolated serving environment is selected"}, "errors": errors,
                "vllm_executable": str(executable), "status": "BLOCKED"}

    for name in SERVING_PACKAGES:
        try:
            installed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed[name] = "MISSING"
            errors.append(f"{name} is not installed")
    for name, expected in SERVING_STACK.items():
        if installed[name] != "MISSING" and Version(installed[name]).base_version != expected:
            errors.append(f"Serving stack mismatch: {name}={installed[name]}, expected {expected}")
    if installed["vllm"] != "MISSING":
        distribution = metadata.distribution("vllm")
        for requirement_text in distribution.requires or []:
            requirement = Requirement(requirement_text)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            try:
                current = metadata.version(requirement.name)
                if not requirement.specifier.contains(current):
                    errors.append(f"vLLM requires {requirement}; installed {requirement.name}={current}")
            except metadata.PackageNotFoundError:
                errors.append(f"Missing vLLM dependency: {requirement}")
        source_path = distribution.locate_file("vllm/model_executor/models/qwen2_5_vl.py")
        source = source_path.read_text(encoding="utf-8") if source_path.is_file() else ""
        if not all(token in source for token in ("SupportsLoRA", "hf_to_vllm_mapper",
                                                 '"model.language_model.": "language_model.model."')):
            errors.append("Installed Qwen2.5-VL implementation lacks the inspected multimodal LoRA/key mapper")

    modules = {}
    for package, module_name in (("torch", "torch"), ("torchvision", "torchvision"),
                                 ("transformers", "transformers"), ("vllm", "vllm"),
                                 ("httpx", "httpx"), ("Pillow", "PIL")):
        if installed.get(package) == "MISSING":
            continue
        try:
            modules[package] = importlib.import_module(module_name)
        except (ImportError, OSError, RuntimeError) as exc:
            errors.append(f"Cannot import serving dependency {module_name}: {exc}")
    transformers = modules.get("transformers")
    if transformers is not None:
        for symbol in ("Qwen2_5_VLForConditionalGeneration", "AutoProcessor"):
            try:
                getattr(transformers, symbol)
            except (ImportError, AttributeError, RuntimeError, OSError) as exc:
                errors.append(f"Transformers cannot import {symbol}: {exc}")

    cuda = {"available": False, "device": None, "verification": "not checked"}
    torch = modules.get("torch")
    if torch is not None:
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        if cuda_version != "12.8":
            errors.append(f"PyTorch must use CUDA 12.8 (cu128); torch.version.cuda={cuda_version!r}")
        try:
            cuda["available"] = bool(torch.cuda.is_available())
            if not cuda["available"]:
                errors.append("PyTorch cannot access CUDA; GPU runtime/device access is unavailable")
            else:
                count = torch.cuda.device_count()
                cuda["device"] = torch.cuda.get_device_name(0)
                if count != 1:
                    errors.append(f"Expected one visible serving GPU, found {count}")
                if "l4" not in cuda["device"].lower():
                    errors.append(f"Expected an NVIDIA L4 serving GPU, found {cuda['device']}")
                cuda["verification"] = "PyTorch CUDA device access only; vLLM kernels and adapter loading unverified"
        except Exception as exc:
            errors.append(f"PyTorch CUDA device access failed: {exc}")
            cuda["verification"] = "device access failed; vLLM kernels and adapter loading unverified"
    executable = Path(sys.executable).parent / "vllm"
    if not executable.is_file():
        errors.append(f"vLLM executable is missing from the active environment: {executable}")
    return {"versions": installed, "cuda": cuda, "errors": errors, "vllm_executable": str(executable),
            "status": "BLOCKED" if errors else "preflight_passed_vllm_kernels_and_adapter_unverified"}


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mode", choices=("base", "adapter"), default="adapter")
    cli.add_argument("--check", action="store_true", help="Check serving imports and basic CUDA access; do not start a server")
    cli.add_argument("--port", type=int, default=int(os.getenv("VLLM_PORT", "8000")))
    cli.add_argument("--max-model-len", type=int, default=int(os.getenv("VLLM_MAX_MODEL_LEN", "2048")))
    cli.add_argument("--max-num-seqs", type=int, default=int(os.getenv("VLLM_MAX_NUM_SEQS", "1")))
    cli.add_argument("--gpu-memory-utilization", type=float, default=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.75")))
    cli.add_argument("--max-lora-rank", type=int, choices=(8, 16, 32, 64), default=8)
    cli.add_argument("--min-pixels", type=int)
    cli.add_argument("--max-pixels", type=int)
    cli.add_argument("--image-limit", type=int, default=1)
    cli.add_argument("--dtype", choices=("bfloat16", "half", "auto"), default="bfloat16")
    cli.add_argument("--base-served-model", default=BASE_SERVED_MODEL)
    cli.add_argument("--adapter-served-model", default=ADAPTER_SERVED_MODEL)
    return cli


def build_command(args, executable):
    if (not 1 <= args.port <= 65535 or args.max_model_len < 1 or args.max_num_seqs < 1
            or args.image_limit < 1 or not math.isfinite(args.gpu_memory_utilization)
            or not 0 < args.gpu_memory_utilization < 1):
        raise ValueError("Invalid server limits or memory utilization")
    if (not args.base_served_model.strip() or not args.adapter_served_model.strip()
            or args.adapter_served_model in {args.base_served_model, BASE_MODEL_ID}):
        raise ValueError("Base and adapter served model names must be distinct")
    adapter = resolve_adapter(max_rank=args.max_lora_rank) if args.mode == "adapter" else None
    min_pixels = args.min_pixels if args.min_pixels is not None else (adapter["min_pixels"] if adapter else 50176)
    max_pixels = args.max_pixels if args.max_pixels is not None else (adapter["max_pixels"] if adapter else 200704)
    if not 0 < min_pixels <= max_pixels:
        raise ValueError("Image pixel bounds must satisfy 0 < min_pixels <= max_pixels")
    command = [str(executable), "serve", BASE_MODEL_ID, "--host", "127.0.0.1", "--port", str(args.port),
               "--served-model-name", args.base_served_model, "--tensor-parallel-size", "1",
               "--dtype", args.dtype, "--gpu-memory-utilization", str(args.gpu_memory_utilization),
               "--max-model-len", str(args.max_model_len), "--max-num-seqs", str(args.max_num_seqs),
               "--max-num-batched-tokens", str(args.max_model_len),
               "--limit-mm-per-prompt", json.dumps({"image": args.image_limit, "video": 0}),
               "--mm-processor-kwargs", json.dumps({"min_pixels": min_pixels, "max_pixels": max_pixels}),
               "--enforce-eager"]
    if adapter:
        command += ["--enable-lora", "--max-loras", "1", "--max-lora-rank", str(args.max_lora_rank),
                    "--lora-modules", json.dumps({"name": args.adapter_served_model, "path": str(adapter["path"]),
                                                  "base_model_name": args.base_served_model}),
                    "--chat-template", str(adapter["template"])]
    # Optional service-local token; never an OpenAI API key. Keep it out of printed argv.
    return command


def main():
    args = parser().parse_args()
    try:
        environment = check_environment()
        print(json.dumps(environment, indent=2), flush=True)
        if environment["errors"]:
            raise RuntimeError("Serving blocked by dependency/version incompatibility; do not force package replacements")
        if args.check:
            return
        command = build_command(args, environment["vllm_executable"])
        print(f"Starting explicit {args.mode} mode on 127.0.0.1:{args.port}; no fallback on failure", flush=True)
        # The CLI accepts VLLM_API_KEY from its environment; no key appears in logs.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.chdir(ROOT)
        os.execv(command[0], command)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
