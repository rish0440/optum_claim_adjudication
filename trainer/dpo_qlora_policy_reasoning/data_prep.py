"""Canonical preference preparation, validation and training split access."""


# Annotation / cases
import copy
import hashlib
import json
import math
import random
from datetime import timedelta
from decimal import Decimal

from src.utils.calculator import adjudicate_invoice
from src.components.agentic_workflow.contracts import CandidateSelections, FactSelections, AgentSafetyError
from src.components.agentic_workflow.policy_adjudicator import PolicyAdjudicationAgent
from src.components.agentic_workflow.reasoning_reviewer import ReasoningReviewAgent
from src.components.inference.vlm_client import BASE_MODEL_ID
from src.components.rag.indexer import source_documents
from src.components.rag.resolver import PolicyRuleResolver
from src.schema import InvoiceGroundTruth, InvoiceLine
from src.utils.paths import ROOT

SEED = 20240901
SPLITS = ("train", "validation", "test")
CONTEXT_LIMIT = 2048
PREFERENCE_REASON = "Canonical policy applicability and exact scoped references match the verified decision; no model-computed financial values."


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(compact(value).encode("utf-8")).hexdigest()


def length_estimate(record):
    maximum_bytes = max(len(compact(record["prompt"]+record[key]).encode("utf-8")) for key in ("chosen", "rejected"))
    return math.ceil(maximum_bytes/2) + 128


def source_hashes():
    """Hash relocated sources under the original manifest identity keys."""
    locations = {
        'src/agents/contracts.py':
            'src/components/agentic_workflow/contracts.py',
        'src/agents/policy_adjudicator.py':
            'src/components/agentic_workflow/policy_adjudicator.py',
        'src/agents/reasoning_reviewer.py':
            'src/components/agentic_workflow/reasoning_reviewer.py',
        'src/prompts/adjudicator.txt':
            'src/components/prompts/adjudicator.txt',
        'src/prompts/reasoning.txt':
            'src/components/prompts/reasoning.txt',
        'src/inference/reasoning_client.py':
            'src/components/inference/reasoning_client.py',
        'src/adjudication/calculator.py':
            'src/utils/calculator.py',
        'src/rag/resolver.py':
            'src/components/rag/resolver.py',
        'src/orchestration/nodes/copilot.py':
            'src/components/agentic_workflow/nodes/copilot.py',
        'src/trainer/preference_optimization/annotation/cases.py':
            'trainer/dpo_qlora_policy_reasoning/data_prep.py',
        'src/trainer/preference_optimization/annotation/generate.py':
            'trainer/dpo_qlora_policy_reasoning/data_prep.py',
        'src/trainer/preference_optimization/annotation/validate.py':
            'trainer/dpo_qlora_policy_reasoning/data_prep.py',
    }
    return {name: hashlib.sha256((ROOT / location).read_bytes()).hexdigest()
            for name, location in locations.items()}


def preferred(schema, data):
    """Canonical label for the RAW selection contract, not processed artifacts."""
    rows = []
    for row in data["items"]:
        if schema is CandidateSelections:
            candidates = row["candidates"]
            candidate = candidates[0] if candidates else None
            if not data["date_applicable"]:
                status, basis, slot = "REVIEW_REQUIRED", "DATE_OUTSIDE", None
            elif candidate is None:
                status, basis, slot = "REVIEW_REQUIRED", "NO_EVIDENCE", None
            elif candidate["kind"] == "exclusion":
                status, basis, slot = "EXCLUDED", "RULE_MATCH", candidate["index"]
            elif row["context_required"]:
                status, basis, slot = "REVIEW_REQUIRED", "CONDITIONS_UNCERTAIN", candidate["index"]
            else:
                status, basis, slot = "COVERED", "RULE_MATCH", candidate["index"]
            rows.append(dict(item_index=row["item_index"], candidate_index=slot, suggested_status=status, basis=basis))
        elif schema is FactSelections:
            rows.append(dict(item_index=row["item_index"], status=row["facts"][0]["status"],
                fact_indices=[fact["index"] for fact in row["facts"]], concerns=row["required_concerns"]))
        else:
            raise ValueError("Unsupported runtime schema")
    return {"items": rows}


class AnnotationClient:
    """Capture the exact system/user messages without constructing an HTTP client."""
    def __init__(self, response=None):
        self.response = response

    def generate(self, instruction, data, schema, *, deadline):
        self.data = copy.deepcopy(data)
        self.messages = [{"role": "system", "content": instruction},
                         {"role": "user", "content": compact(data)}]
        raw = self.response if self.response is not None else preferred(schema, data)
        self.output = schema.model_validate(raw)
        return self.output, {"annotation_source": "offline_canonical_selection"}


def scenario_specs(policy, seed=SEED):
    """Group by underlying rule before assigning splits or generating amounts.

    Room's normal and missing-evidence variants stay together in train. Every
    other category group is isolated; both agent roles inherit the case split.
    """
    rng = random.Random(seed)
    rules = [*policy.coverage_rules, *policy.exclusion_rules[:13]]
    specs = [dict(case_id=f"synthetic-{rule.rule_id}", family=f"category:{rule.rule_id}",
                  rule_id=rule.rule_id, mode="canonical", prior_disagreement=index % 4 == 0)
             for index, rule in enumerate(rules)]
    specs.extend([
        dict(case_id="synthetic-room-evidence-missing", family="category:COV-001", rule_id="COV-001", mode="missing", prior_disagreement=False),
        dict(case_id="synthetic-unverified-imaging-category", family="unverified-category", rule_id=None, mode="similar", prior_disagreement=False),
    ])
    # Exercise dated-policy review without borrowing any held-out claim.
    for spec in specs:
        if spec["rule_id"] == "COV-023":
            spec["mode"] = "date_outside"
        if spec["rule_id"] == "EXC-004":
            spec["mode"] = "excluded_subtype"
    groups = sorted({s["family"] for s in specs if s["family"] != "category:COV-001"})
    rng.shuffle(groups)
    assignments = {group: "test" if i < 5 else "validation" if i < 10 else "train" for i, group in enumerate(groups)}
    assignments["category:COV-001"] = "train"
    by_id = {r.rule_id: r for r in rules}
    for index, spec in enumerate(specs):
        spec["split"] = assignments[spec["family"]]
        rule = by_id.get(spec["rule_id"])
        spec["item"] = ("Unverified imaging service" if rule is None else
                        "Medicines health supplement" if spec["mode"] == "excluded_subtype" else rule.category)
        spec["date"] = (policy.effective_to + timedelta(days=1) if spec["mode"] == "date_outside"
                        else policy.effective_from + timedelta(days=rng.randrange((policy.effective_to-policy.effective_from).days+1))).isoformat()
        cap = getattr(rule, "maximum_reimbursement", None)
        quantity = rng.randint(1, 3)
        limit = cap * quantity if cap and rule.limit_type != "per_claim" else cap
        amount = (limit + Decimal(rng.randint(50, 650)) if limit and index % 2 == 0 else
                  limit * Decimal("0.7") if limit else Decimal(rng.randint(201, 1901)))
        # Two distinct rows within one synthetic case enable wrong-item references.
        spec["lines"] = [{"item": spec["item"], "quantity": quantity, "rate": None, "amount": str(amount)},
                         {"item": spec["item"], "quantity": 1, "rate": None, "amount": str(Decimal(rng.randint(71, 199)))}]
    return specs


def prepare_case(spec, policy):
    resolver = PolicyRuleResolver(policy)
    docs = {d["rule_id"]: {**d, "score": None, "origin": "offline_canonical_fixture"} for d in source_documents(policy)}
    lines = [InvoiceLine.model_validate(row) for row in spec["lines"]]
    invoice = InvoiceGroundTruth(invoice_id=spec["case_id"], image_file="synthetic-no-image",
        invoice_date=spec["date"], lines=lines, billed_total=sum((x.amount for x in lines), Decimal(0)), rate_column_visible=False)
    evidence, resolutions = {}, {}
    for index, line in enumerate(lines):
        applicable_id = resolver.resolve(line.item).get("rule_id")
        # Irrelevant canonical retrieval candidates are inputs to the existing
        # filtering code; they must never become model-facing applicable rules.
        irrelevant = next(d for d in docs.values() if d["rule_id"] != applicable_id and d["kind"] == "exclusion")
        entries = ([] if spec["mode"] == "missing" else
                   [docs[applicable_id], irrelevant] if applicable_id else [docs["COV-007"], irrelevant])
        if index % 2:
            entries = list(reversed(entries))
        evidence[index] = entries
        resolutions[index] = resolver.resolve(line.item, entries) if entries else {
            "status": "manual_review", "rule_id": None, "review_reason": "Missing verified policy evidence"}
    return invoice, evidence, resolutions


def agent1(invoice, evidence, resolutions, policy, response=None):
    client = AnnotationClient(response)
    suggestions, metadata = PolicyAdjudicationAgent(client).suggest(invoice, evidence, resolutions,
        deadline=math.inf, policy=policy)
    return client, suggestions, metadata


def calculate(invoice, resolutions, suggestions, metadata, policy):
    """Mirror only the Phase 7 handoff; financial math is exclusively reused."""
    by_id = {r.rule_id: r for r in policy.coverage_rules + policy.exclusion_rules}
    pairs = []
    for index, line in enumerate(invoice.lines):
        resolution = resolutions[index]
        if index in metadata["canonical_review_items"] and resolution.get("status") == "covered":
            resolution = {**resolution, "status": "manual_review", "rule_id": None,
                          "review_reason": suggestions.items[index].reason}
        pairs.append({"line": line, "resolution": resolution, "rule": by_id.get(resolution.get("rule_id"))})
    return adjudicate_invoice(invoice, policy=policy, resolved_pairs=pairs)


def rejected_selection(role, chosen, data):
    bad = copy.deepcopy(chosen)
    row = bad["items"][0]
    if role == "agent_1":
        if row["suggested_status"] == "COVERED":
            row.update(suggested_status="EXCLUDED", basis="NO_EVIDENCE")
            error = "unsupported_exclusion"
        elif row["suggested_status"] == "EXCLUDED":
            row.update(suggested_status="COVERED", basis="RULE_MATCH")
            error = "exclusion_ignored"
        elif row["basis"] == "CONDITIONS_UNCERTAIN":
            row.update(suggested_status="COVERED", basis="RULE_MATCH")
            error = "unverified_clinical_approval"
        else:
            row.update(suggested_status="EXCLUDED", basis="NO_EVIDENCE")
            error = "unsupported_no_evidence_denial"
    else:
        if row["concerns"] and row["status"] != "manual_review":
            # A real concern on row 0 is not a reason to flag another item.
            bad["items"][1]["concerns"] = row["concerns"][:1]
            error = "unnecessary_wrong_item_concern"
        elif len(row["concerns"]) > 1:
            row["concerns"] = data["items"][1]["required_concerns"][:1]
            error = "wrong_item_concern"
        elif row["concerns"]:
            row["concerns"] = []
            error = "missing_required_concern"
        elif row["status"] == "partial":
            row["status"] = "approved"
            error = "contradictory_status"
        else:
            row["fact_indices"] = [f["index"] for f in data["items"][1]["facts"]]
            error = "wrong_item_fact"
    return bad, error


def make_pair(role, client, spec, policy_ids, claim):
    chosen = client.output.model_dump()
    rejected, error = rejected_selection(role, chosen, client.data)
    record = {"prompt": client.messages, "chosen": [{"role": "assistant", "content": compact(chosen)}],
              "rejected": [{"role": "assistant", "content": compact(rejected)}]}
    # Conservative heuristic only, NOT a tokenizer count or guaranteed bound.
    estimate = length_estimate(record)
    if estimate > CONTEXT_LIMIT:
        raise ValueError("Offline context heuristic exceeded; flag for manual review")
    annotation = {"pair_id": f"{spec['case_id']}-{role}", "agent_role": role,
        "scenario_family": spec["family"], "synthetic_case_id": spec["case_id"], "scenario": spec["mode"],
        "source_policy_ids": sorted(policy_ids), "data_split": spec["split"],
        "chosen_preference_reason": PREFERENCE_REASON,
        "rejected_error_type": error, "case": spec, "deterministic_claim": claim.model_dump(mode="json"),
        "offline_length_estimate": estimate, "validation_status": "pending_automatic_validation",
        "human_verified": False, "record_sha256": fingerprint(record)}
    return record, annotation


def build_pairs(policy, seed=SEED):
    output, flagged = [], []
    for spec in scenario_specs(policy, seed):
        try:
            invoice, evidence, resolutions = prepare_case(spec, policy)
            first, suggestions, metadata = agent1(invoice, evidence, resolutions, policy)
            bad, _ = rejected_selection("agent_1", first.output.model_dump(), first.data)
            _, repaired, bad_metadata = agent1(invoice, evidence, resolutions, policy, bad)
            claim = calculate(invoice, resolutions, suggestions, metadata, policy)
            if calculate(invoice, resolutions, repaired, bad_metadata, policy) != claim:
                raise ValueError("Advisory disagreement changed deterministic financial decisions")
            second = AnnotationClient()
            ReasoningReviewAgent(second).review(invoice, evidence, suggestions, claim, deadline=math.inf,
                policy=policy, agent_metadata=bad_metadata if spec["prior_disagreement"] else metadata)
            ids = {d["rule_id"] for entries in evidence.values() for d in entries}
            pairs = [make_pair(role, client, spec, ids, claim) for role, client in (("agent_1", first), ("agent_2", second))]
            output.extend(pairs)
        except (ValueError, AgentSafetyError) as exc:
            flagged.append({"synthetic_case_id": spec["case_id"], "category": type(exc).__name__,
                            "reason": str(exc), "validation_status": "manual_review_required"})
    return output, flagged


# Annotation / generate
import argparse
import csv
import hashlib
import json

from src.utils.policy import load_policy
from src.components.inference.vlm_client import BASE_MODEL_ID
from src.utils.paths import ROOT, project_path


def generate_dataset(directory="data/policy_reasoning_dpo_qlora", seed=SEED):
    policy = load_policy(ROOT/"data/assignment_dataset/policy.json")
    pairs, flagged = build_pairs(policy, seed)
    for record, annotation in pairs:
        annotation["generation_seed"] = seed
        annotation["validation_status"] = validate_record(record, annotation, policy)
    counts = validate_records(pairs, policy)
    directory = project_path(directory)
    names = [f"{split}.jsonl" for split in SPLITS] + ["manifest.json", "annotation_review.csv"]
    if any((directory/name).exists() for name in names):
        raise FileExistsError("Dataset already exists; choose a new --output-dir (existing files are never overwritten)")
    directory.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (directory/f"{split}.jsonl").write_text("".join(compact(record)+"\n" for record, a in pairs if a["data_split"] == split), encoding="utf-8")
    manifest = {"schema_version": 1, "base_model_id": BASE_MODEL_ID, "served_model": "qwen-base",
        "model_verification": "existing serving configuration and user-verified deployment; no server queried",
        "chat_template": "Qwen2.5-VL processor/tokenizer chat template; system/user prompt, assistant-only chosen/rejected; do not pre-render or append EOS manually",
        "generation_seed": seed, "target_pairs": 80, "target_splits": {"train": 60, "validation": 10, "test": 10},
        "pair_count": len(pairs), "counts_by_split_role": counts, "policy_source": "data/assignment_dataset/policy.json",
        "policy_sha256": hashlib.sha256((ROOT/"data/assignment_dataset/policy.json").read_bytes()).hexdigest(),
        "source_sha256": source_hashes(), "human_verified": False,
        "retrieval_provenance": "synthetic canonical retrieval fixtures; no Milvus query or embedding",
        "flagged_cases": flagged, "annotations": [a for _, a in pairs]}
    (directory/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    fields = ("pair_id", "role", "scenario", "preferred_response_summary", "rejected_response_summary",
              "why_chosen_is_preferred", "applicable_canonical_policy", "validation_status", "human_verified")
    with (directory/"annotation_review.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record, a in pairs:
            writer.writerow(dict(pair_id=a["pair_id"], role=a["agent_role"], scenario=a["scenario_family"]+" / "+a["scenario"],
                preferred_response_summary=record["chosen"][0]["content"], rejected_response_summary=record["rejected"][0]["content"],
                why_chosen_is_preferred=a["chosen_preference_reason"]+" Rejected error: "+a["rejected_error_type"],
                applicable_canonical_policy=", ".join(sorted({row["rule_id"] for row in a["deterministic_claim"]["decisions"] if row["rule_id"]}
                    | ({a["case"]["rule_id"]} if a["case"]["rule_id"] and a["case"]["mode"] not in {"missing", "date_outside"} else set()))),
                validation_status=a["validation_status"], human_verified=False))
    return {"pairs": len(pairs), "counts_by_split_role": counts, "flagged_cases": flagged,
            "human_verified": False, "output_dir": str(directory)}


def generate_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/policy_reasoning_dpo_qlora")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(json.dumps(generate_dataset(args.output_dir, args.seed), indent=2))


# Annotation / validate
import argparse
import json
import hashlib
import csv
from collections import Counter

from src.utils.policy import load_policy
from src.components.agentic_workflow.contracts import AgentSafetyError, CandidateSelections, FactSelections
from src.components.agentic_workflow.reasoning_reviewer import ReasoningReviewAgent
from src.components.inference.reasoning_client import _unique_object, _reject_constant
from src.components.inference.vlm_client import BASE_MODEL_ID
from src.utils.paths import ROOT, project_path


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
    from .provenance import verify_provenance
    directory = project_path(directory)
    manifest = strict_json((directory/"manifest.json").read_text(encoding="utf-8"))
    policy = load_policy(ROOT/"data/assignment_dataset/policy.json")
    verify_provenance(manifest)
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


def validate_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/policy_reasoning_dpo_qlora")
    args = parser.parse_args()
    print(json.dumps(validate_dataset(args.dataset_dir), indent=2))


# Train / dataset
import hashlib
from src.utils.policy import load_policy
from src.utils.paths import ROOT, project_path


def load_preferences(directory):
    from .provenance import verify_provenance
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
    counts = validate_records(entries, load_policy(ROOT/"data/assignment_dataset/policy.json"))
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


def main():
    import sys
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=('generate', 'validate'))
    args = cli.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    original = sys.argv
    try:
        sys.argv = [original[0], *remaining]
        commands = {'generate': 'generate_main', 'validate': 'validate_main'}
        globals()[commands[args.action]]()
    finally:
        sys.argv = original



if __name__ == "__main__":
    main()
