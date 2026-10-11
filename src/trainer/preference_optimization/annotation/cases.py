"""Offline canonical annotations, prompt capture and deterministic case replay.

No HTTP client, tokenizer, embedding model, image or training model is loaded.
The annotation client is deliberately offline, never a production agent fallback.
"""
import copy
import hashlib
import json
import math
import random
from datetime import timedelta
from decimal import Decimal

from src.adjudication.calculator import adjudicate_invoice
from src.agents.contracts import CandidateSelections, FactSelections, AgentSafetyError
from src.agents.policy_adjudicator import PolicyAdjudicationAgent
from src.agents.reasoning_reviewer import ReasoningReviewAgent
from src.inference.vlm_client import BASE_MODEL_ID
from src.rag.indexer import source_documents
from src.rag.resolver import PolicyRuleResolver
from src.schemas import InvoiceGroundTruth, InvoiceLine
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
    paths = ("src/agents/contracts.py", "src/agents/policy_adjudicator.py",
             "src/agents/reasoning_reviewer.py", "src/prompts/adjudicator.txt",
             "src/prompts/reasoning.txt", "src/inference/reasoning_client.py",
             "src/adjudication/calculator.py", "src/rag/resolver.py",
             "src/orchestration/nodes/copilot.py",
             "src/trainer/preference_optimization/annotation/cases.py",
             "src/trainer/preference_optimization/annotation/generate.py",
             "src/trainer/preference_optimization/annotation/validate.py")
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}


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
