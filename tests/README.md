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

The suite now also covers scoped agent contracts, two-model routing, browser/PDF
integration, preference annotation/training/evaluation and CORD extraction.
See [docs/OPERATIONS.md](../docs/OPERATIONS.md) for focused regression commands.
No current pass count is claimed by the repository cleanup; run the suite manually.
Original invoice tests need locally retained, authorized image fixtures. Some
historical preference provenance checks can report source drift after Phase 9D.

Tests use temporary files for rendering/packaging and mocked remote integrations.
They do not train a model, start vLLM, download model weights or regenerate the
repository's ten synthetic invoices. Live Milvus and GPU inference are separate
integration checks and are not claimed by this suite.
