"""Pinned compatibility for the trained preference dataset after source relocation.

The original manifest stays immutable. Policy, schemas, prompts, agent logic,
calculator, resolver and annotation implementation must still match exactly.
Only the pinned routing/trace revisions, import/path relocations and consolidation below are
recognized; annotation labels and dataset manifests remain unchanged.
Callers MUST subsequently replay their selected records with validate_records:
that verifies exact runtime prompts, canonical selections and Decimal decisions.
The data preparation validator replays every record after this source check.
"""
import hashlib

from src.utils.paths import ROOT
from trainer.dpo_qlora_policy_reasoning.data_prep import source_hashes


RUNTIME_TRANSITIONS = {
    'src/agents/contracts.py': (
        'df3422d63aa8a89a93cc1fd32c62197301c9b0a433d7676eb62db4246f02d5d7',
        '2e16425d49a04253724b189ce95fbc40e3266ed5278c86b46e07579848674d5d',
        "Reviewed source consolidation and module/path relocation; exact canonical "
        "prompt, selection and Decimal decision replay remain required",
    ),
    'src/agents/policy_adjudicator.py': (
        '1024659462cd0274934862a61fc85121085f84e5c2c726327daa14a07f057be5',
        '420658fd57b48efa07766c91e23ed230964cdb68cfa5edaf342a3b422df52056',
        "Reviewed source consolidation and module/path relocation; exact canonical "
        "prompt, selection and Decimal decision replay remain required",
    ),
    'src/agents/reasoning_reviewer.py': (
        '65c35655e92d091dbb2ddd4c344d6eb57bf29e6cd1df60a1798c22eafb2d616d',
        '624ae750ec49961aa208a3b32950a262627e39532d31bfe544ba3193205742b2',
        "Reviewed source consolidation and module/path relocation; exact canonical "
        "prompt, selection and Decimal decision replay remain required",
    ),
    'src/inference/reasoning_client.py': (
        'b1dea523d31434e63e7a9fc46bd574069384523eab9c3058b65bb88672520fa6',
        '647eb477890c2d9987967ff2e699441e39159440057122f989c5c1bdedaa9bd3',
        "Reviewed source consolidation and module/path relocation; exact canonical "
        "prompt, selection and Decimal decision replay remain required",
    ),
    'src/adjudication/calculator.py': (
        'b13548d8f401ad22fb5d5fc5e72c422cd999b9f600f3efdd916686c480b7efd9',
        '443665c543628c06a25e1607fb9b49dda03a0697ce35fe70bcd50cfacbf5397c',
        "Reviewed source consolidation and module/path relocation; exact canonical "
        "prompt, selection and Decimal decision replay remain required",
    ),
    'src/rag/resolver.py': (
        'ac1f93bfadf91a3c4c2501d326f49890214cd1b526cb2181617d41593a753c96',
        'c5d377d552fd57a6a050d286dedb3660d6c2fe37400f87814cf4064269275dbb',
        "Reviewed source consolidation and module/path relocation; exact canonical "
        "prompt, selection and Decimal decision replay remain required",
    ),
    'src/orchestration/nodes/copilot.py': (
        'bb78959baa96c4ba238bb262b11e7afddd481dc634afac5db8cca7472ca94a3e',
        '232ea0a8fb251ad77bf3f25dc2cc65169a1c45e877db9cd251860bb6039dbb88',
        "Reviewed source consolidation and module/path relocation; exact canonical "
        "prompt, selection and Decimal decision replay remain required",
    ),
    'src/trainer/preference_optimization/annotation/cases.py': (
        'bf3803924981790b28d1ee64587bf90f1175962a25be019a4394928a29088f7b',
        '5034d92fdaf5f99fafe6cb92de649c94705277df43adffb7805c73fc458705b6',
        "Reviewed source consolidation and module/path relocation; exact canonical "
        "prompt, selection and Decimal decision replay remain required",
    ),
    'src/trainer/preference_optimization/annotation/generate.py': (
        '7a7c16b3048b0809700609f3b0e27f118ae729a7778fb9e74ff4e4798989f20c',
        '5034d92fdaf5f99fafe6cb92de649c94705277df43adffb7805c73fc458705b6',
        "Reviewed source consolidation and module/path relocation; exact canonical "
        "prompt, selection and Decimal decision replay remain required",
    ),
    'src/trainer/preference_optimization/annotation/validate.py': (
        '8c74a95aea3f233d486be234b9cd8f758c30e97727deeae08c1ba1ee01ae3b84',
        '5034d92fdaf5f99fafe6cb92de649c94705277df43adffb7805c73fc458705b6',
        "Reviewed source consolidation and module/path relocation; exact canonical "
        "prompt, selection and Decimal decision replay remain required",
    ),
}


def verify_provenance(manifest):
    """Fail on policy drift, missing fingerprints or any unreviewed revision."""
    policy_hash = hashlib.sha256((ROOT / "data/assignment_dataset/policy.json").read_bytes()).hexdigest()
    if manifest.get("policy_sha256") != policy_hash:
        raise ValueError("Phase 9A canonical policy fingerprint mismatch; do not relabel data")
    recorded, current = manifest.get("source_sha256"), source_hashes()
    if not isinstance(recorded, dict) or set(recorded) != set(current):
        raise ValueError("Phase 9A source fingerprint set mismatch")
    changes = {}
    for path, actual in current.items():
        expected = recorded[path]
        if expected == actual:
            continue
        transition = RUNTIME_TRANSITIONS.get(path)
        if transition is None or (expected, actual) != transition[:2]:
            raise ValueError(f"Phase 9A unreviewed source fingerprint mismatch: {path}")
        changes[path] = {"recorded_sha256": expected, "current_sha256": actual,
                         "reason": transition[2]}
    return {"status": "compatible_runtime_revision" if changes else "exact",
            "policy_sha256": policy_hash, "source_changes": changes,
            "record_replay": "required; selected records checked by validate_records"}
