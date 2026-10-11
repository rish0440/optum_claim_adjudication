"""Read only train/validation; reuse Phase 9A validation without opening test."""
import hashlib
from src.adjudication.policy import load_policy
from src.trainer.preference_optimization.annotation.provenance import verify_provenance
from src.trainer.preference_optimization.annotation.validate import strict_json, validate_records
from src.utils.paths import ROOT, project_path


def load_preferences(directory):
    directory = project_path(directory).resolve()
    manifest = strict_json((directory/"manifest.json").read_text(encoding="utf-8"))
    provenance = verify_provenance(manifest)
    if manifest["pair_count"] != 80 or manifest["flagged_cases"]:
        raise ValueError("Expected the verified complete 80-pair Phase 9A dataset")
    # Original annotation provenance is VL; training explicitly uses the NEW
    # text backbone. Do not rewrite the original dataset or its manifest.
    if manifest["base_model_id"] != "Qwen/Qwen2.5-VL-3B-Instruct":
        raise ValueError("Unexpected preference-data source model")
    family_splits, case_splits = {}, {}
    for a in manifest["annotations"]:
        for mapping, key in ((family_splits, a["scenario_family"]), (case_splits, a["synthetic_case_id"])):
            if key in mapping and mapping[key] != a["data_split"]:
                raise ValueError("Manifest indicates cross-split case/family leakage")
            mapping[key] = a["data_split"]
    data, entries, hashes = {}, [], {}
    for split, expected in (("train", 60), ("validation", 10)):
        path = directory/f"{split}.jsonl"
        raw = path.read_bytes()
        data[split] = [strict_json(line) for line in raw.decode("utf-8").splitlines()]
        metadata = [a for a in manifest["annotations"] if a["data_split"] == split]
        if len(data[split]) != expected or len(metadata) != expected:
            raise ValueError(f"Expected exactly {expected} {split} pairs")
        entries.extend(zip(data[split], metadata))
        hashes[split] = hashlib.sha256(raw).hexdigest()
    counts = validate_records(entries, load_policy(ROOT/"data/policy.json"))
    if counts["train"] != {"agent_1": 30, "agent_2": 30} or counts["validation"] != {"agent_1": 5, "agent_2": 5}:
        raise ValueError("Unexpected role balance")
    audit = {"source_model_id": manifest["base_model_id"], "train_pairs": 60, "validation_pairs": 10,
             "counts_by_role": {s: counts[s] for s in data}, "split_sha256": hashes,
             "manifest_sha256": hashlib.sha256((directory/"manifest.json").read_bytes()).hexdigest(),
             "test_accessed": False, "source_provenance": provenance,
             "annotation_validator": "Phase 9A validate_records on train/validation only"}
    return data, audit


def serialize_and_audit(record, tokenizer, max_length, *, template_fn=None, tokenize_fn=None):
    # Import TRL lazily; CPU loader tests need no Torch or tokenizer downloads.
    if template_fn is None:
        from trl.data_utils import apply_chat_template
        template_fn = apply_chat_template
    if tokenize_fn is None:
        from trl import DPOTrainer
        tokenize_fn = DPOTrainer.tokenize_row
    rendered = template_fn(record, tokenizer)
    prompt = tokenizer.apply_chat_template(record["prompt"], tokenize=False, add_generation_prompt=True)
    if rendered["prompt"] != prompt:
        raise ValueError("Text Qwen chat template changed the prompt/completion boundary")
    for key in ("chosen", "rejected"):
        full = tokenizer.apply_chat_template(record["prompt"]+record[key], tokenize=False)
        if prompt+rendered[key] != full or record[key][0]["content"] not in rendered[key]:
            raise ValueError("Assistant JSON is missing or leaked into the prompt")
    tokens = tokenize_fn(rendered, processing_class=tokenizer, max_prompt_length=None,
                         max_completion_length=None, add_special_tokens=False)
    if not tokens["prompt_input_ids"] or tokens["chosen_input_ids"] == tokens["rejected_input_ids"]:
        raise ValueError("Empty prompt or indistinguishable preference token sequences")
    # Audit exactly TRL 0.24 tokenization, including its appended completion EOS.
    for key in ("chosen", "rejected"):
        ids = tokens[f"{key}_input_ids"]
        if not ids or ids[-1] != tokenizer.eos_token_id or len(tokens["prompt_input_ids"])+len(ids) > max_length:
            raise ValueError("Preference pair exceeds configured sequence budget or has no completion EOS; no truncation allowed")
        content = tokenizer.decode(ids, skip_special_tokens=True).strip()
        if content != record[key][0]["content"].strip():
            raise ValueError("Tokenized completion does not decode to the exact assistant JSON")
    return rendered, tokens


def prepare_preferences(data, tokenizer, max_length):
    from datasets import Dataset
    if not tokenizer.chat_template or tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Qwen tokenizer requires chat template, padding and EOS tokens")
    datasets = {}
    for split, records in data.items():
        rendered = [serialize_and_audit(row, tokenizer, max_length)[0] for row in records]
        # Standard TRL explicit prompt/chosen/rejected strings, already rendered
        # with the text model's template. Its template pass is then a no-op.
        datasets[split] = Dataset.from_list(rendered)
    return datasets
