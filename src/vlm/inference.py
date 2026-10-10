"""Qwen2.5-VL invoice-image -> CSV inference (GPU, Colab Phase 3).

Examples:
  python -m src.vlm.inference --image data/synthetic/validation/images/invoice_00004.jpg
  python -m src.vlm.inference --image sample.jpg --adapter models/qwen25-vl-invoice-lora

For Phase 3 this runs inside Colab, NOT on the Windows FastAPI laptop.
Phase 5 remote inference over HTTP is implemented separately in remote_client.py.
"""

import argparse
import gc
from pathlib import Path

from .training_data import INSTRUCTION

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28


def load_model(model_name: str = DEFAULT_MODEL, adapter: str | None = None,
               max_pixels: int = MAX_PIXELS):
    """Load base or LoRA-equipped model in NF4 4-bit onto one NVIDIA GPU."""
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for Phase 3 VLM inference. Use Google Colab.")
    processor = AutoProcessor.from_pretrained(
        adapter if adapter and (Path(adapter) / "preprocessor_config.json").exists() else model_name,
        min_pixels=MIN_PIXELS, max_pixels=max_pixels,
    )
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True,
                                     bnb_4bit_compute_dtype=torch.float16)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, quantization_config=quantization,
        device_map={"": 0}, dtype=torch.float16, attn_implementation="sdpa",
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    return model, processor


def extract_csv(model, processor, image_path: str | Path, max_new_tokens: int = 400) -> str:
    """Run actual multimodal generation, not simulated labels or policy outputs."""
    import torch
    from PIL import Image

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    prompt = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": INSTRUCTION},
    ]}]
    template = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[template], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                do_sample=False, use_cache=True)
    token_ids = output[0][inputs["input_ids"].shape[1]:]
    text = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
    del inputs, output
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model, processor = load_model(args.model, args.adapter, args.max_pixels)
    result = extract_csv(model, processor, args.image, args.max_new_tokens)
    print(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    del model, processor
    gc.collect()


if __name__ == "__main__":
    main()
