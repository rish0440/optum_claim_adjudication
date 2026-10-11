"""Small report writer: publish only real, complete paired evaluations."""
import csv
import json


def comparison_rows(metrics):
    directions = {"unsupported_reference_count": False, "wrong_item_reference_count": False,
                  "unsupported_approval_count": False, "missing_required_concern_count": False,
                  "unsupported_review_count": False, "unsupported_approval_rate": False,
                  "wrong_item_evidence_rate": False, "generation_failures": False,
                  "average_generation_seconds": False}
    rows = []
    for group in ("overall", "agent_1", "agent_2"):
        left = metrics["baseline"]["overall"] if group == "overall" else metrics["baseline"]["by_role"].get(group, {})
        right = metrics["dpo"]["overall"] if group == "overall" else metrics["dpo"]["by_role"].get(group, {})
        for name in left:
            if name in {"samples", "preference"}:
                continue
            a, b = left[name], right[name]
            if isinstance(a, dict):
                a, b = a["percent"], b["percent"]
            direction = directions.get(name, True)
            delta = b-a if a is not None and b is not None else None
            verdict = "not measured" if delta is None else "tied" if abs(delta) <= 1e-9 else (
                "improved" if (delta > 0) == direction else "worsened")
            rows.append(dict(group=group, metric=name, baseline=a, dpo=b, delta=delta, dpo_result=verdict))
        if left:
            for name in ("accuracy", "mean_margin"):
                a, b = left["preference"][name], right["preference"][name]
                if isinstance(a, dict):
                    a, b = a["percent"], b["percent"]
                delta = b-a if a is not None and b is not None else None
                verdict = "not measured" if delta is None else "tied" if abs(delta) <= 1e-9 else "improved" if delta > 0 else "worsened"
                rows.append(dict(group=group, metric="raw_preference_"+name, baseline=a, dpo=b, delta=delta, dpo_result=verdict))
    return rows


def write_reports(directory, predictions, scores, metrics, metadata):
    ids = [p["pair_id"] for p in predictions["baseline"]]
    if (metadata.get("status") != "completed" or len(ids) != metadata["dataset"]["pairs"] or
            len(ids) != len(set(ids)) or ids != [p["pair_id"] for p in predictions["dpo"]] or
            ids != [s["pair_id"] for s in scores]):
        raise ValueError("Refuse fabricated/incomplete comparison report")
    def write_json(name, value):
        (directory/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    for model in ("baseline", "dpo"):
        (directory/(model+"_predictions.jsonl")).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n" for row in predictions[model]), encoding="utf-8")
    write_json("pairwise_scores.json", scores)
    write_json("metrics.json", metrics)
    rows = comparison_rows(metrics)
    with (directory/"comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group", "metric", "baseline", "dpo", "delta", "dpo_result"])
        writer.writeheader()
        writer.writerows(rows)
    reviews = []
    for baseline, dpo in zip(predictions["baseline"], predictions["dpo"]):
        reviews.append({"pair_id": baseline["pair_id"], "agent_role": baseline["agent_role"],
            "prompt_summary": baseline["scenario_family"], "canonical_policy_ids": baseline["source_policy_ids"],
            "verified_expected": baseline["expected_items"],
            "baseline_response": baseline["raw_output_sanitized"], "dpo_response": dpo["raw_output_sanitized"],
            "baseline_status": baseline["assessment"]["correctness_status"],
            "dpo_status": dpo["assessment"]["correctness_status"],
            "baseline_failures": baseline["assessment"]["failures"], "dpo_failures": dpo["assessment"]["failures"],
            "human_review_status": "PENDING"})
    (directory/"human_review.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in reviews), encoding="utf-8")
    def number(value):
        return "N/A" if value is None else f"{value:.3f}"
    cases = metadata["dataset"].get("underlying_cases", len({p["scenario_family"] for p in predictions["baseline"]}))
    role_counts = {role: sum(p["agent_role"] == role for p in predictions["baseline"]) for role in ("agent_1", "agent_2")}
    lines = ["# Independent baseline vs DPO evaluation", "",
        f"Both modes: `{metadata['model_id']}`. DPO adapter SHA256: `{metadata['adapter']['adapter_sha256']}`.",
        f"Checkpoint: `{metadata['adapter']['adapter']}`.",
        f"Split: **test**, {len(ids)} pairs ({role_counts}), from {cases} underlying synthetic cases. Test SHA256: `{metadata['dataset']['test_sha256']}`.",
        "Identical base object, tokenizer, chat template, prompts and greedy settings. Adapter state is audited for every mode.", "",
        "## Generation and raw preference ranking", "",
        "Percentages are percentages of explicit denominators in metrics.json; counts are counts. Missing denominators are N/A.", "",
        "eligible_completion means a nontruncated schema-valid answer with row-owned references, not approval or semantic correctness. Safety rates count affected prompts; reference/approval counts count offending selections.", "",
        "| Group | Metric | Baseline | DPO | Change | DPO result |",
        "|---|---|---:|---:|---:|---|"]
    lines += [f"| {r['group']} | {r['metric']} | {number(r['baseline'])} | {number(r['dpo'])} | {number(r['delta'])} | {r['dpo_result']} |" for r in rows]
    lines += ["", "## Preference outcome counts", ""]
    for mode in ("baseline", "dpo"):
        p = metrics[mode]["overall"]["preference"]
        lines.append(f"- {mode}: correct={p['correct']}, incorrect={p['incorrect']}, tied={p['tied']}, scoring failures={p['failures']}.")
    lines += ["", "Raw preference accuracy ranks summed chosen/rejected completion log probabilities; it is neither DPO reference-relative reward accuracy nor claim accuracy.",
        "Per-scenario results and their small denominators are in metrics.json.", "", "## Representative actual generation failures", ""]
    failures = [(mode, p) for mode in ("baseline", "dpo") for p in predictions[mode] if p["assessment"]["failures"]]
    lines += [f"- {mode}, {p['pair_id']}: {', '.join(p['assessment']['failures'])}. See sanitized output in predictions JSONL." for mode, p in failures[:6]] or ["No automatic generation-contract failures observed in this run."]
    lines += ["", "## Runtime", "",
        f"Total evaluation time: {metadata.get('total_seconds', 0):.2f}s. Peak tracked CUDA allocation: {metadata.get('peak_cuda_allocated_gib', 0):.2f} GiB.",
        "Per-generation latency and pairwise scoring time are recorded separately.", "", "## Limitations and conclusion", "",
        f"Only {len(ids)} synthetic preference pairs ({cases} correlated case groups) were evaluated. These labels are automatically canonical-validated, not independently human verified.",
        "Negatives are deliberately constructed single-field/scoped-reference errors, often easy; they do not represent the full difficulty of clinical reasoning or real claim review.",
        "Both modes use the same NF4 quantized base; this comparison does not measure full-precision baseline behavior.",
        "Agent 2 receives the same fixed verified decisions/Agent 1 context in both modes; this is independent reasoning evaluation, not a new end-to-end deployment.",
        "No free-text explanation quality is claimed because raw agent contracts contain selections, not prose. Semantics are scored only where canonical annotation replay establishes the expected decision.",
        "Training did not record a base-weight revision/hash; evaluation records its resolved base revision/config hash. Confirm the local base snapshot matches the training snapshot before drawing conclusions.",
        "Metric changes above report improved/worsened/tied independently. No overall superiority or deployment decision is inferred. Do not tune prompts, checkpoints or parameters from these test results."]
    (directory/"comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json("metadata.json", metadata)  # last file; parent publishes directory only after completion
