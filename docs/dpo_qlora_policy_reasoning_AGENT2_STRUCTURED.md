# Agent 2 constrained-generation diagnostic

Root cause: the Phase 9C HF `Backend.generate()` used unconstrained greedy
decoding, unlike production `TextReasoningClient`, which sends a strict vLLM
JSON schema. Saved generations contain forbidden totals and echoed input facts.
They also contain missing facts and cross-item references: these are semantic
errors, not merely JSON formatting. The fix does not alter agent prompts,
contracts, labels, weights or the original evaluation implementation/results.

The standalone runner uses the existing HTTP transport and vLLM 0.11.2's
`response_format: json_schema` mechanism. Schema is derived unchanged from
`FactSelections`: only items, item_index, status, fact_indices and concerns;
no totals, policy documents or amounts. It does NOT narrow integer choices to
gold/reference IDs or select the right status. Missing/wrong facts and concerns
remain possible and are explicitly rejected by the original runtime validator.
Requests contain only the original system/user messages; chosen/rejected labels
never go to generation. No correction, retries, mock fallback or default answers.
Token-budget truncation remains a failed generation even with structured output.

## Manual CPU preparation

```bash
cd /home/rishabh_mishra0440/optum_claim_adjudication/optum_claim_adjudication
source ../.venv/bin/activate
python -B -m unittest tests.test_reasoning_routing -v
python -u -m trainer.dpo_qlora_policy_reasoning.evaluate_agent --check
python -u -m trainer.dpo_qlora_policy_reasoning.evaluate_agent --print-server-command > /tmp/optum-agent2-eval-server.sh
cat /tmp/optum-agent2-eval-server.sh
```

`--check` is CPU-only: registered validation cases and pointer/checksum verification,
no network or model loading. `--print-server-command` prints only a command;
it never starts a process. No new dependency is required. Runtime needs the
already-installed vLLM 0.11.2 in `~/.venv-vllm`; no changes to either environment.

## Temporary text server (manual, separate from Phase 8)

Port 8002 uses new names `reasoning-base` and `reasoning-dpo`, with the verified
adapter and saved text tokenizer. Both names share one identical BF16 text base.
The baseline name requests no LoRA; the DPO name requests that adapter. Model-card
root/parent/path are checked before inference. Server `/tokenize` output must
match between both names and the saved training tokenizer/chat template.

**One L4 cannot safely host the existing 0.75–0.80-memory invoice server plus
this 0.75-memory text server concurrently.** Manually stop the existing server
first (Ctrl-C in its terminal). Existing server config, launcher, files and
weights are unchanged. Restart it using your usual command after diagnostics.
Do not launch both at those memory settings or kill processes automatically.

In a separate terminal, after freeing the GPU:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 bash /tmp/optum-agent2-eval-server.sh
```

This command uses the absolute `~/.venv-vllm/bin/python` printed by the helper.
Offline mode requires the already-cached TEXT model. Source inspection confirms
the chosen vLLM 0.11.2 CLI/model-card/schema APIs; kernels, adapter loading and
actual constraint compilation remain unverified until you start it. Unsupported
schema/adapter loading fails; there is no base-only or unconstrained fallback.

## Smoke and comparison (original ../.venv terminal)

```bash
python -u -m trainer.dpo_qlora_policy_reasoning.evaluate_agent --check-server
python -u -m trainer.dpo_qlora_policy_reasoning.evaluate_agent --split validation --limit 1 --mode both
python -u -m trainer.dpo_qlora_policy_reasoning.evaluate_agent --split validation --mode both
```

All generation settings, token budget (512 by default), schema and prompts are
identical for baseline and DPO. This diagnostic uses BF16 vLLM, unlike the original
NF4 Transformers evaluator; do not attribute differences from the OLD results
solely to schema constraints or DPO. Within the new comparison both modes share
the same precision/base. It measures generation only, not preference ranking.

Default validation is implementation debugging, not new generalization evidence.
The observed test requires explicit acknowledgment:

```bash
python -u -m trainer.dpo_qlora_policy_reasoning.evaluate_agent --split test --acknowledge-observed-test --mode both
```

That is a diagnostic RETEST, never a fresh independent evaluation. Do not alter
prompts, fine-tune, choose checkpoints or relabel from these test failures. New
independently held-out cases are still needed to estimate generalization.

Reports publish in `reports/dpo/agent2_structured/<printed-run-id>/`:
baseline/dpo prediction JSONL, metrics.json, metadata.json, comparison.md and
human_review.jsonl. Infrastructure failure leaves only an uncompleted `.pending`
directory with failure diagnostics. Original Phase 9C artifacts are untouched.

```bash
cat reports/dpo/agent2_structured/<printed-run-id>/comparison.md
python -m json.tool reports/dpo/agent2_structured/<printed-run-id>/metrics.json
cat reports/dpo/agent2_structured/<printed-run-id>/human_review.jsonl
```

Each prediction contains actual sanitized output, mode, measured generation
tokens/latency, structured-decoding flag, separate JSON/schema/scoped/semantic
checks, runtime-validator acceptance and allowed/selected reference diagnostics.
Validation replay feeds the actual generated selections into the original
ReasoningReviewAgent against canonical fixtures; it never substitutes its labels
as generations. Exact Decimal decisions/totals are checked for preservation.

Phase 9D is not declared ready automatically. Require non-test approved, partial,
excluded and genuine-review cases to pass the actual runtime validator without
changing amounts, inspect failures, then obtain fresh independent evaluation.
Schema constraints alone cannot fix wrong-item fact choices, missing required
facts/concerns or incorrect statuses. Further model improvement may be needed,
but no training change is justified from this observed test set alone.

No tests, inference, installations, training or services were executed during
code preparation.

## Final NEW medical-evidence manual-review smoke

`--manual-review-smoke` builds exactly one deterministic, standalone synthetic
claim from `trainer/dpo_qlora_policy_reasoning/evaluate_agent.py`; it never opens train/validation/test JSONL.
Medicines (COV-005), dated 2024-06-17, are billed at INR 1475.25 with no supplied
prescription or established medical-treatment relationship. Existing canonical
rules and the Decimal calculator must leave that amount pending and final
reimbursement null. The actual Agent 2 input carries its verified decision fact
0, canonical policy fact 1 and POLICY_CONTEXT_MISSING concern 0. Expected output
is retained separately for validation, never added to generation messages or
JSON-schema constraints. This is a smoke diagnostic, not independent evidence
of generalization.

With the existing text server already running on port 8002:

```bash
source ../.venv/bin/activate
python -B -m unittest tests.test_reasoning_routing -v
python -u -m trainer.dpo_qlora_policy_reasoning.evaluate_agent --manual-review-smoke --check
python -u -m trainer.dpo_qlora_policy_reasoning.evaluate_agent --manual-review-smoke --mode both --endpoint http://127.0.0.1:8002 --output-dir reports/dpo/agent2_manual_review
```

Both `reasoning-base` and `reasoning-dpo` receive identical original messages,
generic strict FactSelections schema and token budget. The existing semantic
scorer and original runtime review validator check every real generation.
New run reports contain generated JSON, expected references and authoritative
pending amounts (metadata.json), semantic/runtime acceptance and errors
(prediction JSONL), plus human_review.jsonl. Existing datasets, weights and
evaluation reports are preserved. No command above was executed by Codex.
