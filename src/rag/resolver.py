"""Resolve policy candidates using rule identity, exclusions and specificity.

Similarity alone is not proof of coverage. Known categories/approved synonyms
are resolved deterministically against policy.json; unknown terms require review.
This module deliberately NEVER calculates claim payouts (Phase 7).
"""

import re

from src.policy import load_policy

# A small transparent normalization glossary, NOT learned from Optum test images.
# Keys refer to exact IDs from the supplied policy and are validated at runtime.
# Keep deliberately conservative to avoid false denials/approvals.
ALIASES: dict[str, tuple[str, ...]] = {
    "COV-001": ("hospital room charge", "ward charge", "bed charge"),
    "COV-002": ("icu", "intensive care unit charge"),
    "COV-003": ("doctor consultation", "physician consultation", "consultation fee"),
    "COV-004": ("specialist visit", "specialist consultation fee"),
    "COV-005": ("medication", "medicine charge", "pharmacy medicine"),
    "COV-006": ("diagnostic test", "diagnostic investigation"),
    "COV-007": ("mri", "magnetic resonance imaging"),
    "COV-008": ("ct", "computed tomography", "cat scan"),
    "COV-009": ("xray", "x ray", "radiograph"),
    "COV-010": ("blood test", "blood work", "cbc", "lipid profile", "complete blood count"),
    "COV-011": ("sonography", "ultrasound scan"),
    "COV-012": ("electrocardiogram", "ecg test"),
    "COV-019": ("surgical glove", "sterile glove", "medical consumable"),
    "COV-020": ("oxygen therapy charge", "oxygen supply"),
    "EXC-001": ("personal care kit", "toiletries", "toiletry kit", "grooming kit"),
    "EXC-004": ("health supplement", "nutrition supplement", "vitamin supplement"),
    "EXC-006": ("diet meal", "special diet meal", "patient diet meal"),
    "EXC-007": ("television charge", "tv charge", "room television rental"),
    "EXC-008": ("phone charge", "telephone usage"),
    "EXC-009": ("wifi charge", "wi fi charge", "internet access charge"),
    "EXC-011": ("processing fee", "administration fee"),
    "EXC-012": ("registration fee", "admission registration fee"),
}


def normalize_item(text: str) -> str:
    """Lowercase, separate punctuation, and standardize regular plurals."""
    words = re.findall(r"[a-z0-9]+", text.lower().replace("x-ray", "x ray"))
    result = []
    for word in words:
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            word = word[:-1]
        result.append(word)
    return " ".join(result)


def _is_match(item_normalized: str, phrase_normalized: str) -> bool:
    """Exact names or a named test followed by extra specificity, e.g. (CBC)."""
    if item_normalized == phrase_normalized:
        return True
    # Lab test variants: Blood Test (CBC), Blood Test (Lipid Profile), etc.
    return phrase_normalized == "blood test" and item_normalized.startswith("blood test ")


class PolicyRuleResolver:
    """Ground rule selection in the original JSON policy, never the vector scores."""

    def __init__(self, policy=None):
        self.policy = policy or load_policy()
        self.by_id = {r.rule_id: r for r in
                      self.policy.coverage_rules + self.policy.exclusion_rules}
        for rule_id in ALIASES:
            if rule_id not in self.by_id:
                raise ValueError(f"Alias references missing policy rule: {rule_id}")

    def resolve(self, item: str, hits: list[dict] | None = None) -> dict:
        if not item or not item.strip():
            raise ValueError("Item name cannot be blank")
        normalized = normalize_item(item)
        hits = hits or []
        known_ids = set(self.by_id)
        # Only expose citations that actually occur in the source policy.
        candidates = [hit for hit in hits if hit.get("rule_id") in known_ids]
        matches = []
        for rule in self.by_id.values():
            category_name = normalize_item(rule.category)
            if _is_match(normalized, category_name):
                matches.append((rule, "exact_category"))
            elif any(_is_match(normalized, normalize_item(term)) for term in
                     ALIASES.get(rule.rule_id, ())):
                matches.append((rule, "approved_alias"))
        if matches:
            # Exclusions override coverage. Within coverage, exact/specific outranks
            # a synonym/general category. No silent semantic approval.
            matches.sort(key=lambda value: (
                0 if value[0].rule_id.startswith("EXC-") else 1,
                0 if value[1] == "exact_category" else 1,
                -len(normalize_item(value[0].category)),
                value[0].rule_id,
            ))
            rule, match_method = matches[0]
            kind = "exclusion" if rule.rule_id.startswith("EXC-") else "coverage"
            # A medical supply can fall under COV-019 or COV-024 depending on
            # inpatient usage; record this interpretation for later adjudication.
            review_note = ("Surgical gloves mapped to general medical consumables; "
                           "verify treatment context if policy conditions require it."
                           if normalized in {"surgical glove", "sterile glove"} else None)
            return self._make_result(item, rule, kind, match_method, candidates, review_note)

        return {
            "item": item,
            "status": "manual_review",
            "rule_id": None,
            "category": None,
            "match_method": "unverified_semantic_candidates",
            "limit_type": None,
            "maximum_reimbursement": None,
            "unit": None,
            "conditions": None,
            "source": "data/policy.json",
            "evidence": None,
            "review_reason": "No verified category or synonym match; semantic similarity alone is insufficient.",
            "retrieval_hit": False,
            "candidates": candidates,
        }

    def _make_result(self, item, rule, kind, match_method, candidates, review_note):
        return {
            "item": item,
            "status": "excluded" if kind == "exclusion" else "covered",
            "rule_id": rule.rule_id,
            "category": rule.category,
            "match_method": match_method,
            "limit_type": getattr(rule, "limit_type", None),
            "maximum_reimbursement": (str(rule.maximum_reimbursement)
                                      if getattr(rule, "maximum_reimbursement", None) is not None else None),
            "unit": getattr(rule, "unit", None),
            "conditions": rule.conditions,
            "source": "data/policy.json",
            "evidence": f"{rule.rule_id} — {rule.category}: {rule.conditions}",
            "review_reason": review_note,
            "retrieval_hit": any(hit.get("rule_id") == rule.rule_id for hit in candidates),
            "candidates": candidates,
        }
