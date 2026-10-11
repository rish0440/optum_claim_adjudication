"""Strict CPU validation and replay of synthetic preference annotations."""
import argparse
import json
import hashlib
import csv
from collections import Counter

from src.adjudication.policy import load_policy
from src.agents.contracts import AgentSafetyError, CandidateSelections, FactSelections
from src.agents.reasoning_reviewer import ReasoningReviewAgent
from src.inference.reasoning_client import _unique_object, _reject_constant
from src.inference.vlm_client import BASE_MODEL_ID
from src.utils.paths import ROOT, project_path
from .cases import (AnnotationClient, agent1, calculate, compact, fingerprint, prepare_case,
                    preferred, rejected_selection, scenario_specs, source_hashes, SPLITS,
                    length_estimate, CONTEXT_LIMIT, PREFERENCE_REASON)


def strict_json(text):
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def completion(record, key, schema):
    messages = record[key]
    if not isinstance(messages, list) or len(messages) != 1 or set(messages[0]) != {"role", "content"} or messages[0]["role"] != "assistant":
        raise ValueError(f"{key}: expected one assistant-only completion")
    return schema.model_validate(strict_json(messages[0]["content"]))


def validate_record(record, annotation, policy):
    if set(record) != {"prompt", "chosen", "rejected"}:
        raise ValueError("Preference records require exactly prompt/chosen/rejected")
    messages = record["prompt"]
    if not isinstance(messages, list) or len(messages) != 2 or any(set(m) != {"role", "content"} for m in messages) or [m["role"] for m in messages] != ["system", "user"]:
        raise ValueError("Prompt roles must exactly match runtime system/user messages")
    role = annotation["agent_role"]
    if role not in {"agent_1", "agent_2"}:
        raise ValueError("Unknown agent role")
    schema = CandidateSelections if role == "agent_1" else FactSelections
    chosen, rejected = (completion(record, key, schema) for key in ("chosen", "rejected"))
    if chosen == rejected:
        raise ValueError("Chosen and rejected must differ")
    seed = annotation["generation_seed"]
    # Reject arbitrary edited cases and accidental personal records. We accept
    # only this generator's fully reproducible synthetic case registry.
    registry = {s["case_id"]: s for s in scenario_specs(policy, seed)}
    spec = registry.get(annotation["synthetic_case_id"])
    if spec is None or annotation["case"] != spec or annotation["scenario_family"] != spec["family"] or annotation["data_split"] != spec["split"]:
        raise ValueError("Unrecognized case, personal data or inconsistent split/family")
    if annotation["pair_id"] != f"{spec['case_id']}-{role}":
        raise ValueError("Pair identity mismatch")
    if annotation["chosen_preference_reason"] != PREFERENCE_REASON or annotation["scenario"] != spec["mode"]:
        raise ValueError("Unrecognized audit prose or scenario")
    if annotation["offline_length_estimate"] != length_estimate(record) or length_estimate(record) > CONTEXT_LIMIT:
        raise ValueError("Invalid/off-limit offline length estimate")
    invoice, evidence, resolutions = prepare_case(spec, policy)
    canonical_ids = {r.rule_id for r in policy.coverage_rules + policy.exclusion_rules}
    evidence_ids = {doc["rule_id"] for entries in evidence.values() for doc in entries}
    if set(annotation["source_policy_ids"]) != evidence_ids or not evidence_ids <= canonical_ids:
        raise ValueError("Unknown or incorrect canonical policy reference")
    first, suggestions, metadata = agent1(invoice, evidence, resolutions, policy)
    bad1, _ = rejected_selection("agent_1", first.output.model_dump(), first.data)
    _, _, bad_metadata = agent1(invoice, evidence, resolutions, policy, bad1)
    claim = calculate(invoice, resolutions, suggestions, metadata, policy)
    if annotation["deterministic_claim"] != claim.model_dump(mode="json"):
        raise ValueError("Fabricated financial result or decision")
    if role == "agent_1":
        captured = first
        # Candidate slots in BOTH completions must resolve or be null. Incorrect
        # status/basis is a negative label, not permission to invent a candidate.
        for output in (chosen, rejected):
            if [r.item_index for r in output.items] != list(range(len(invoice.lines))):
                raise ValueError("Wrong invoice row identity")
            for row, context in zip(output.items, captured.data["items"]):
                if row.candidate_index is not None and row.candidate_index not in {c["index"] for c in context["candidates"]}:
                    raise ValueError("Unresolvable candidate reference")
        _, _, chosen_metadata = agent1(invoice, evidence, resolutions, policy, chosen.model_dump())
        if chosen_metadata["disagreement_items"]:
            raise ValueError("Chosen Agent 1 suggestion lacks canonical support")
        _, _, rejected_metadata = agent1(invoice, evidence, resolutions, policy, rejected.model_dump())
        if not rejected_metadata["disagreement_items"]:
            raise ValueError("Rejected Agent 1 output has no demonstrated reasoning error")
    else:
        captured = AnnotationClient(chosen.model_dump())
        reviewer = ReasoningReviewAgent(captured)
        before = claim.model_dump(mode="json")
        reviewer.review(invoice, evidence, suggestions, claim, deadline=float("inf"), policy=policy,
            agent_metadata=bad_metadata if spec["prior_disagreement"] else metadata)
        if claim.model_dump(mode="json") != before:
            raise ValueError("Agent review modified calculator result")
        # Wrong-item negatives may select a real fact from another row, but no
        # completion is allowed to introduce a nonexistent fact/concern ID.
        all_facts = {f["index"] for row in captured.data["items"] for f in row["facts"]}
        all_concerns = {c["index"] for row in captured.data["items"] for c in row["concern_facts"]}
        if any(not set(row.fact_indices) <= all_facts or not set(row.concerns) <= all_concerns for row in rejected.items):
            raise ValueError("Fabricated fact/concern reference in negative example")
        negative_client = AnnotationClient(rejected.model_dump())
        try:
            ReasoningReviewAgent(negative_client).review(invoice, evidence, suggestions, claim, deadline=float("inf"),
                policy=policy, agent_metadata=bad_metadata if spec["prior_disagreement"] else metadata)
        except AgentSafetyError as exc:
            observed = exc.diagnostics["category"]
        else:
            raise ValueError("Rejected Agent 2 response is valid; preference unsupported")
        expected_categories = {"wrong_item_fact": "unknown_or_missing_fact", "contradictory_status": "contradictory_status",
                               "missing_required_concern": "unsupported_or_missing_concern",
                               "wrong_item_concern": "unsupported_or_missing_concern",
                               "unnecessary_wrong_item_concern": "unsupported_or_missing_concern"}
        if expected_categories.get(annotation["rejected_error_type"]) != observed:
            raise ValueError("Documented negative criterion differs from actual validator rejection")
    if messages != captured.messages:
        raise ValueError("Prompt differs from runtime input, policy conditions or verified amounts")
    if chosen.model_dump() != preferred(schema, captured.data):
        raise ValueError("Chosen answer differs from supported scoped selection")
    expected_bad, expected_error = rejected_selection(role, chosen.model_dump(), captured.data)
    if rejected.model_dump() != expected_bad or annotation["rejected_error_type"] != expected_error:
        raise ValueError("Unsupported negative annotation or wrong preference criterion")
    if annotation["record_sha256"] != fingerprint(record):
        raise ValueError("Record checksum mismatch")
    if annotation.get("human_verified") is not False:
        raise ValueError("Synthetic labels have not been independently human verified")
    return "automatically_validated_human_review_pending"


def validate_records(entries, policy):
    pairs, prompts, cases, families = set(), {}, {}, {}
    counts = {split: Counter() for split in SPLITS}
    for record, annotation in entries:
        pair_id, split = annotation["pair_id"], annotation["data_split"]
        if split not in SPLITS or pair_id in pairs:
            raise ValueError("Duplicate pair ID or unknown split")
        pairs.add(pair_id)
        prompt_hash = fingerprint(record["prompt"])
        if prompt_hash in prompts:
            raise ValueError("Duplicate prompt/preference pair")
        prompts[prompt_hash] = split
        for mapping, value in ((cases, annotation["synthetic_case_id"]), (families, annotation["scenario_family"])):
            if value in mapping and mapping[value] != split:
                raise ValueError("Cross-split underlying case/family leakage")
            mapping[value] = split
        validate_record(record, annotation, policy)
        counts[split][annotation["agent_role"]] += 1
    if any(c["agent_1"] != c["agent_2"] for c in counts.values()):
        raise ValueError("Both roles must remain paired within each split")
    return {split: dict(count) for split, count in counts.items()}


def validate_dataset(directory):
    directory = project_path(directory)
    manifest = strict_json((directory/"manifest.json").read_text(encoding="utf-8"))
    policy = load_policy(ROOT/"data/policy.json")
    if manifest["policy_sha256"] != hashlib.sha256((ROOT/"data/policy.json").read_bytes()).hexdigest() or manifest["source_sha256"] != source_hashes():
        raise ValueError("Policy or agent contracts changed; regenerate offline annotations")
    if manifest["base_model_id"] != BASE_MODEL_ID or manifest["human_verified"] is not False:
        raise ValueError("Incorrect backbone or unverified human-verification claim")
    annotations = manifest["annotations"]
    entries = []
    for split in SPLITS:
        records = [strict_json(line) for line in (directory/f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        metadata = [a for a in annotations if a["data_split"] == split]
        if len(metadata) != len(records):
            raise ValueError("JSONL and manifest counts differ")
        entries.extend(zip(records, metadata))
    if len(entries) != len(annotations) or any(a["generation_seed"] != manifest["generation_seed"] for a in annotations):
        raise ValueError("Manifest split/seed mismatch")
    counts = validate_records(entries, policy)
    if counts != manifest["counts_by_split_role"] or manifest["pair_count"] != len(entries):
        raise ValueError("Manifest counts do not match validated records")
    if len(entries) + 2*len(manifest["flagged_cases"]) != manifest["target_pairs"]:
        raise ValueError("Unreported pair shortfall")
    expected_cases = {s["case_id"] for s in scenario_specs(policy, manifest["generation_seed"])}
    actual_cases = {a["synthetic_case_id"] for _, a in entries}
    flagged = manifest["flagged_cases"]
    flagged_ids = {f["synthetic_case_id"] for f in flagged}
    if (len(flagged_ids) != len(flagged) or actual_cases & flagged_ids or actual_cases | flagged_ids != expected_cases
            or any(f["validation_status"] != "manual_review_required" for f in flagged)):
        raise ValueError("Missing, duplicate or incorrectly flagged cases")
    roles_by_case = {}
    for _, annotation in entries:
        roles_by_case.setdefault(annotation["synthetic_case_id"], set()).add(annotation["agent_role"])
        if annotation["validation_status"] != "automatically_validated_human_review_pending":
            raise ValueError("Manifest annotation status is inconsistent")
    if any(roles != {"agent_1", "agent_2"} for roles in roles_by_case.values()):
        raise ValueError("A case lost one of its agent preference pairs")
    with (directory/"annotation_review.csv").open(newline="", encoding="utf-8") as stream:
        review_rows = list(csv.DictReader(stream))
    by_id = {a["pair_id"]: (record, a) for record, a in entries}
    if len(review_rows) != len(entries) or len({r["pair_id"] for r in review_rows}) != len(entries):
        raise ValueError("Review report count/duplicate mismatch")
    for row in review_rows:
        if row["pair_id"] not in by_id:
            raise ValueError("Review report contains an unknown pair")
        record, annotation = by_id[row["pair_id"]]
        if (row["preferred_response_summary"] != record["chosen"][0]["content"] or
                row["rejected_response_summary"] != record["rejected"][0]["content"] or
                row["role"] != annotation["agent_role"] or row["human_verified"] != "False" or
                row["validation_status"] != annotation["validation_status"]):
            raise ValueError("Review report differs from actual annotations")
    return {"status": "PASS", "pairs": len(entries), "counts_by_split_role": counts,
            "manual_review_cases": len(manifest["flagged_cases"]), "human_verified": False,
            "token_counts": "offline heuristic only; tokenizer verification deferred"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/dpo")
    args = parser.parse_args()
    print(json.dumps(validate_dataset(args.dataset_dir), indent=2))


if __name__ == "__main__":
    main()
