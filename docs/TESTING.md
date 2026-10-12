# Essential tests

Run from the repository root using the existing application/training environment:

```bash
source ../.venv/bin/activate
OPTUM_REASONING_BACKEND=phase8 python -B -m unittest discover -s tests -v
```

The suite runs offline with mocked external services. It does not load model
backbones, train on a GPU or prove live vLLM/Milvus integration. Reference invoice
fixtures and the existing preference dataset must be available locally.

| Test module | Purpose |
| --- | --- |
| `test_claim_graph.py` | Compiled workflow, real calculations, evidence handling and bounded failures |
| `test_claim_web.py` | Image upload, invoice dates, job results, downloads and PDF reports |
| `test_reasoning_routing.py` | Separate extraction/reasoning endpoints and verified adapter identity |
| `test_agent1_integration.py` | Canonical policy selection, citations and approval safeguards |
| `test_vllm_inference.py` | Image requests, CSV validation, checkpoint resolution and serving launchers |
| `test_policy_rag.py` | Policy indexing, retrieval and safe coverage/exclusion matching |
| `test_adjudication.py` | Decimal totals, coverage caps, copay, policy dates and manual review |
| `test_adapter_checkpoints.py` | Dataset split boundaries and safe successful-checkpoint selection |
| `test_cord_training.py` | Training split isolation, label validation, completion masks and checkpoints |
| `test_dpo_training.py` | Preference provenance, split isolation, completion boundaries and checkpoints |

`helpers.py` contains only the two synthetic receipt fixtures required by the
CORD training tests. Historical phase, migration, dataset-generation and detailed
evaluation suites were removed; essential retrieval and adjudication tests were
renamed by responsibility. Production application and training code is unchanged.
