"""Manual, isolated held-out evaluation of existing and CORD vision adapters."""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

from src.trainer.sft.train.train_qlora import versions, write_json
from src.utils.paths import project_path
from .evaluation_data import hashlib_prompt, load_test, resolve_checkpoints
from .evaluation_metrics import score
from .schema import fingerprint, sha, write_jsonl
from .train import MODEL_ID


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset", choices=("cord", "optum"), default="cord")
    cli.add_argument("--dataset-root", type=Path, default=Path("data/cord_200"))
    cli.add_argument("--baseline-root", type=Path)
    cli.add_argument("--cord-root", type=Path, default=Path("models/qwen25_vl_cord_sft"))
    cli.add_argument("--output-dir", type=Path, help="Defaults to the selected dataset's isolated report directory")
    cli.add_argument("--min-pixels", type=int, default=50176)
    cli.add_argument("--max-pixels", type=int, default=200704)
    cli.add_argument("--max-context-length", type=int, default=2048)
    cli.add_argument("--max-new-tokens", type=int, default=1024)
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--local-files-only", action="store_true")
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="CPU artifact/fingerprint checks, no model loading")
    mode.add_argument("--smoke", action="store_true", help="First held-out image on BOTH adapters; clearly partial report")
    mode.add_argument("--inspect", type=Path, help="Inspect an existing completed report and representative cases")
    return cli


def common_settings(args, instruction, revision):
    if (args.min_pixels < 1 or args.max_pixels < args.min_pixels or args.max_new_tokens < 1
            or args.max_context_length <= args.max_new_tokens or args.seed < 0):
        raise ValueError("Invalid image/context/generation limits or seed")
    return {"model_id": MODEL_ID, "revision": revision, "instruction": instruction,
            "prompt_sha256": hashlib_prompt(instruction), "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels, "max_context_length": args.max_context_length,
            "max_new_tokens": args.max_new_tokens, "do_sample": False, "num_beams": 1,
            "dtype": "bfloat16", "quantization": "4-bit NF4 double quantization",
            "attention": "sdpa", "seed": args.seed, "local_files_only": args.local_files_only,
            "allow_empty_rates": args.dataset == "optum"}


def assert_adapter_active(model, mode):
    if model.active_adapters != [mode] or mode not in model.peft_config:
        raise RuntimeError("Wrong/missing active PEFT adapter; no base-model fallback is allowed")
    if model.peft_config[mode].base_model_name_or_path != MODEL_ID:
        raise RuntimeError("Active adapter backbone differs from the shared model")


def verify_loaded_weights(model, mode, adapter, expected_sha256):
    """Verify PEFT's loaded tensors against the saved adapter, not just its name."""
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    weights = Path(adapter) / "adapter_model.safetensors"
    if sha(weights) != expected_sha256:
        raise RuntimeError("Adapter weights changed after CPU verification")
    actual = get_peft_model_state_dict(model, adapter_name=mode, save_embedding_layers=False)
    expected = load_file(str(weights), device="cpu")
    if set(actual) != set(expected) or not expected:
        raise RuntimeError("PEFT adapter tensor mapping differs from the verified saved checkpoint")
    if any(not torch.equal(actual[key].detach().float().cpu(), value.float()) for key, value in expected.items()):
        raise RuntimeError("Active model adapter tensors differ from the saved weights")


class AdapterComparison:
    """One quantized base, two small frozen adapters, evaluated sequentially."""
    def __init__(self, adapters, settings):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, set_seed
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise RuntimeError("One visible BF16 CUDA GPU required; activate ../.venv and use CUDA_VISIBLE_DEVICES=0")
        free, total = torch.cuda.mem_get_info(0)
        if free / 2**30 < 18:
            raise RuntimeError("Less than 18 GiB free; manually stop BOTH vLLM servers before evaluation")
        set_seed(settings["seed"])
        torch.cuda.reset_peak_memory_stats(0)
        self.settings = settings
        self.gpu = {"name": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                    "total_gib": total/2**30, "initial_free_gib": free/2**30}
        print("Loading one shared NF4 Qwen-VL base/processor; no text reasoning model", flush=True)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=settings["revision"],
            use_fast=False, min_pixels=settings["min_pixels"], max_pixels=settings["max_pixels"],
            local_files_only=settings["local_files_only"], trust_remote_code=False)
        if not self.processor.chat_template:
            raise RuntimeError("Shared Qwen processor is missing its chat template")
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, revision=settings["revision"],
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
            device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=settings["local_files_only"], trust_remote_code=False)
        if getattr(base.config, "_commit_hash", None) != settings["revision"]:
            raise RuntimeError("Loaded base revision differs from the common evaluation revision")
        self.model = PeftModel.from_pretrained(base, adapters["baseline"]["path"],
                                              adapter_name="baseline", is_trainable=False)
        self.model.load_adapter(adapters["cord_sft"]["path"], adapter_name="cord_sft", is_trainable=False)
        self._freeze_for_inference()
        self.model.config.use_cache = True
        for mode in ("baseline", "cord_sft"):
            verify_loaded_weights(self.model, mode, adapters[mode]["path"], adapters[mode]["adapter_sha256"])
        self.vision_calls = 0
        self.hook = self.model.get_base_model().visual.register_forward_hook(self._vision_used)

    def _vision_used(self, module, inputs, output):
        self.vision_calls += 1

    def _freeze_for_inference(self):
        # Loading another adapter or set_adapter can re-enable active LoRA grads.
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("Evaluation model unexpectedly has trainable parameters")

    def select(self, mode):
        self.model.set_adapter(mode)
        self._freeze_for_inference()
        assert_adapter_active(self.model, mode)

    def generate(self, image_path, instruction):
        """Only the image/instruction enter generation; no labels or reference CSV."""
        import torch
        from PIL import Image
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        # Same Qwen message/image path as the working src.inference.local implementation.
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": instruction}]}]
        template = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[template], images=[image], padding=True,
                                truncation=False, return_tensors="pt")
        prompt_tokens = inputs["input_ids"].shape[1]
        if prompt_tokens + self.settings["max_new_tokens"] > self.settings["max_context_length"]:
            return {"raw_response": "", "error": "context_budget_exceeded_no_truncation",
                    "prompt_tokens": prompt_tokens, "generated_tokens": 0, "truncated": False}
        if any(key not in inputs or inputs[key].numel() == 0 for key in ("pixel_values", "image_grid_thw")):
            raise RuntimeError("Missing actual multimodal image tensors")
        inputs = inputs.to("cuda")
        previous = self.vision_calls
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.settings["max_new_tokens"],
                                         do_sample=False, num_beams=1, use_cache=True)
        tokens = output[0][prompt_tokens:]
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        eos = {value for value in eos if value is not None}
        raw = self.processor.tokenizer.decode(tokens.tolist(), skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)
        stopped = len(tokens) > 0 and tokens[-1].item() in eos
        vision_verified = self.vision_calls > previous
        return {"raw_response": raw, "error": None if vision_verified else "vision_forward_not_verified", "prompt_tokens": prompt_tokens,
                "generated_tokens": len(tokens), "truncated": not stopped,
                "vision_forward_verified": vision_verified}

    def close(self):
        self.hook.remove()
        del self.model, self.processor
        gc.collect()
        import torch
        torch.cuda.empty_cache()


def evaluate_predictions(records, predictions, settings):
    """Check common image/settings identities before any paired scoring."""
    comparisons = []
    for mode in ("baseline", "cord_sft"):
        if len(predictions[mode]) != len(records):
            raise ValueError("Incomplete model comparison; fixed denominators cannot be reduced")
    for index, record in enumerate(records):
        row = {"image": record["image"], "image_path": str(record["image_path"]),
               "ground_truth_csv": record["ground_truth_csv"]}
        for mode in ("baseline", "cord_sft"):
            prediction = predictions[mode][index]
            if (prediction["image"] != record["image"] or prediction["image_sha256"] != record["image_sha256"]
                    or prediction["common_settings_sha256"] != fingerprint(settings) or prediction["mode"] != mode):
                raise ValueError("Model/input/config identity differs between paired evaluations")
            row[mode] = {**prediction, "metrics": score(record["ground_truth_csv"], prediction,
                        allow_empty_rates=settings.get("allow_empty_rates", False))}
        comparisons.append(row)
    return comparisons


def run(args):
    if args.inspect:
        from .evaluation_report import inspect_report
        inspect_report(project_path(args.inspect))
        return None
    if Path(sys.prefix).name != ".venv" or sys.prefix == sys.base_prefix:
        raise RuntimeError("Activate the existing application/training ../.venv")
    adapters, instruction, revision = resolve_checkpoints(
        project_path(args.baseline_root) if args.baseline_root else None, project_path(args.cord_root))
    settings = common_settings(args, instruction, revision)
    if args.dataset == "optum":
        from .optum_evaluation import load_optum
        records, dataset = load_optum()
    else:
        records, dataset = load_test(project_path(args.dataset_root), adapters)
    libraries = versions()
    for package in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "Pillow", "safetensors"):
        if libraries[package] == "MISSING":
            raise RuntimeError(f"Missing existing evaluation dependency: {package}; no installation performed")
    metadata = {"model_id": MODEL_ID, "adapters": adapters, "settings": settings, "dataset_name": args.dataset,
                "common_settings_sha256": fingerprint(settings), "dataset": dataset, "versions": libraries,
                "baseline_training_revision_recorded": bool(adapters["baseline"]["training_config"].get("resolved_model_revision")),
                "interpreter": sys.executable, "environment": sys.prefix}
    if args.preflight:
        print(json.dumps({"status": "CPU_PREFLIGHT_PASS_GPU_UNVERIFIED", **metadata}, indent=2), flush=True)
        return None
    report_root = "reports/cord/optum_final" if args.dataset == "optum" else "reports/cord/evaluation"
    output = project_path(args.output_dir or report_root).resolve()
    if not output.is_relative_to(project_path(report_root).resolve()):
        raise ValueError(f"Evaluation outputs must remain under {report_root}")
    records = records[:1] if args.smoke else records
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    directory = output / run_id
    directory.mkdir(parents=True, exist_ok=False)
    metadata.update(run_id=run_id, smoke=args.smoke, evaluated_documents=len(records), status="running",
                    inputs=[{key: record[key] for key in ("image", "image_sha256", "pixel_sha256")} for record in records])
    write_json(directory / "evaluation_manifest.json", metadata)
    engine, predictions, started = None, {}, time.monotonic()
    try:
        engine = AdapterComparison(adapters, settings)
        metadata.update(gpu=engine.gpu, chat_template_sha256=hashlib_prompt(engine.processor.chat_template))
        for mode in ("baseline", "cord_sft"):
            engine.select(mode)
            predictions[mode] = []
            for index, record in enumerate(records, 1):
                if sha(record["image_path"]) != record["image_sha256"]:
                    raise ValueError("Test image changed during evaluation")
                started_image = time.monotonic()
                try:
                    prediction = engine.generate(record["image_path"], instruction)
                except Exception as exc:
                    prediction = {"raw_response": "", "error": type(exc).__name__, "truncated": False,
                                  "generated_tokens": 0, "prompt_tokens": None}
                    import torch
                    torch.cuda.empty_cache()
                prediction.update(mode=mode, image=record["image"], image_sha256=record["image_sha256"],
                                  pixel_sha256=record["pixel_sha256"],
                                  common_settings_sha256=fingerprint(settings), latency_seconds=time.monotonic()-started_image)
                predictions[mode].append(prediction)
                print(f"{mode} {index}/{len(records)} {record['image']} tokens={prediction['generated_tokens']} "
                      f"truncated={prediction['truncated']} error={prediction['error']}", flush=True)
            write_jsonl(directory / f"{mode}_predictions.jsonl", predictions[mode])
        comparisons = evaluate_predictions(records, predictions, settings)
        import torch
        metadata.update(status="complete_smoke" if args.smoke else "complete",
                        runtime_seconds=time.monotonic()-started,
                        peak_cuda_allocated_gib=torch.cuda.max_memory_allocated(0)/2**30,
                        peak_cuda_reserved_gib=torch.cuda.max_memory_reserved(0)/2**30)
        from .evaluation_report import write_reports
        summary = write_reports(directory, comparisons, metadata)
        if args.dataset == "optum":
            from .optum_evaluation import add_recommendation
            add_recommendation(directory, summary)
        metadata["report_sha256"] = {name: sha(directory / name) for name in (
            "baseline_predictions.jsonl", "cord_sft_predictions.jsonl", "per_image_comparison.csv",
            "per_image_comparison.json", "summary.json", "summary.md")}
        if args.dataset == "optum":
            metadata["report_sha256"]["comparison.csv"] = sha(directory / "comparison.csv")
        write_json(directory / "evaluation_manifest.json", metadata)
        print(f"Report: {directory}", flush=True)
        return directory
    except BaseException as exc:
        metadata.update(status="failed", error_type=type(exc).__name__, runtime_seconds=time.monotonic()-started)
        write_json(directory / "evaluation_manifest.json", metadata)
        print(f"Evaluation failed ({type(exc).__name__}); partial artifacts retained at {directory}", flush=True)
        raise
    finally:
        if engine is not None:
            engine.close()


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
