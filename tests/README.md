# CPU regression tests

- Extraction schema, malformed images and absent invoice rate columns.
- Policy retrieval hits with correct exact rule IDs.
- Coverage versus exclusions and specific-rule precedence.
- Per-day/per-claim caps, co-pay, currency rounding and line-item reconciliation.
- FastAPI upload endpoint and all five held-out invoices.

Run from the repository root using the existing VM environment:

```bash
../.venv/bin/python -B -m unittest discover -s tests -v
```

The seven original test modules contain 66 tests, including two compiled
LangGraph tests enabled when LangGraph is installed. `test_migration.py` adds
eight checks covering legacy imports, real API-to-graph execution, configurable
paths, standalone training bundles, inference import isolation and the corrected
Hugging Face `images` sequence representation. The latter skips if `datasets`
is unavailable. On this VM all 74 tests pass with no skips.

Tests use temporary files for rendering/packaging and mocked remote integrations.
They do not train a model, start vLLM, download model weights or regenerate the
repository's ten synthetic invoices. Live Milvus and GPU inference are separate
integration checks and are not claimed by this suite.
