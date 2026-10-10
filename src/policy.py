"""Read and validate the supplied policy without inventing policy rules."""

import json
from pathlib import Path

from src.schemas import PolicyDocument
from src.settings import get_settings


def load_policy(path: Path | None = None) -> PolicyDocument:
    """Load policy.json and check required fields, rule IDs, and effective dates."""
    policy_path = path or get_settings().paths.policy
    with policy_path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    return PolicyDocument.model_validate(raw)


def summarize_policy(policy: PolicyDocument) -> dict:
    """Return a small inspection summary without modifying the supplied JSON."""
    return {
        "policy_id": policy.policy_id,
        "currency": policy.currency,
        "effective_from": policy.effective_from.isoformat(),
        "effective_to": policy.effective_to.isoformat(),
        "sections": len(policy.policy_sections),
        "coverage_rules": len(policy.coverage_rules),
        "exclusion_rules": len(policy.exclusion_rules),
        "copay_percent": str(policy.reimbursement_rules.copay.rate_percent),
    }
