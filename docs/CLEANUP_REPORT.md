# Final repository cleanup and delivery review

## Inspection baseline — 2026-10-11

- Repository: `/home/rishabh_mishra0440/optum_claim_adjudication/optum_claim_adjudication`.
- Branch: `repo_clean`; latest commit: `beb0298` (`cord dataset addition training`).
- Fetch/push remote: `https://github.com/rish0440/optum_claim_adjudication.git`.
- Initial worktree: only untracked
  `reports/inference/adapter/20261011T150500Z-d19b7f8c/`. Preserved locally;
  the new report ignore rule hides it from ordinary status without removing it.
- Existing module references, launcher settings, requirements and all three
  latest-successful pointers were inspected. No application code, adapter or
  dataset was executed, changed or moved. No Git index/history changes made.

## Retained and organized

| Responsibility | Retained implementation |
| --- | --- |
| Browser/API/PDF | `app/` |
| Extraction and reasoning transport | `src/inference/`, `src/config/` |
| Two real agents and compiled workflow | `src/agents/`, `src/orchestration/` |
| Canonical rules, RAG, Decimal math | `data/policy.json`, `src/rag/`, `src/adjudication/` |
| Training and evaluation | `src/trainer/sft/`, `src/trainer/cord/`, `src/trainer/preference_optimization/` |
| CLI and compatibility imports | `scripts/`, `src/vlm/`, `src/agentic_workflow/` and existing wrappers |
| Fixtures and tests | `data/synthetic/`, `data/dpo/`, `data/evaluation/`, `data/demos/`, `tests/` |
| Environment declarations | All existing `requirements*.txt`, serving constraints, `.env.example` |

`trainer/` and `evaluation/` are documentation indexes, not new implementations.
`notebooks/` already existed; its historical notebook is retained with a warning
against running old installation cells in the working environments. No Python
package, import or CLI path changed.

The previous root README is preserved as `docs/IMPLEMENTATION_HISTORY.md` with
an archive notice. New README supersedes old phase status/ports. Added operations
instructions and updated artifact/test/report documentation.

`deploy/vllm/start_reasoning.sh` preserves the settings read from
`/tmp/optum-agent2-eval-server-40.sh`: text Qwen backbone, saved adapter tokenizer,
reasoning-base / reasoning-dpo, loopback 8002, BF16, context 2048, one sequence,
memory utilization 0.40, generation-config vllm, rank 8 and one LoRA slot.
Only path/interpreter discovery becomes portable, using the existing canonical
successful-pointer/checksum helper. The temporary script remains untouched.
Vision launcher, production inference, agents, policy, calculations and UI are
unchanged. The new launcher has not been syntax-tested or launched.

## Three selected adapters

| Root | Selected run | Weights |
| --- | --- | --- |
| `models/qwen25_vl_sft` | `20261010T174916Z-0b733db4` | `runs/20261010T174916Z-0b733db4/adapter/adapter_model.safetensors` |
| `models/qwen25_3b_reasoning_dpo` | `20261011T074847Z-60ab777d` | `runs/20261011T074847Z-60ab777d/adapter/adapter_model.safetensors` |
| `models/qwen25_vl_cord_sft` | `20261011T134348Z-f1e33111` | `runs/20261011T134348Z-f1e33111/adapter/adapter_model.safetensors` |

Pointers were inspected, not altered. Their recorded hashes are in
`models/README.md`. Deep tensor/hash verification remains a manual command.
`.gitignore` allows only these exact successful runs, explicit adapter artifacts,
their pointers and essential provenance/configuration/metrics. Failed/smoke runs,
periodic checkpoints and pretrained weights stay ignored. The selected small
JSON training histories are essential metadata for current checkpoint contracts;
console `.log` files and other training histories are excluded.

`.gitattributes` assigns LFS to exactly the three selected weight files. **Git
LFS is not installed on this VM. No files were staged or converted to LFS.**
The remote host is GitHub, which supports [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).
Repository-specific permissions, LFS quota/billing and successful authenticated
upload are not verified. Do not stage weights until LFS is installed and its
local filter is active. Never use `git add -f` or bypass an LFS/size failure.

## Proposed index-only exclusions — no local deletion

These commands are proposals for manual review, not actions performed by cleanup.
Ignoring already tracked files does not remove them from the Git index/history.

| Paths | Evidence / action |
| --- | --- |
| `reports/claim_graph/`, `reports/dpo/`, `reports/generated/`, `reports/inference/` | Generated request/evaluation artifacts, not runtime dependencies; remove from tracked deliverable only, retain locally. |
| `reports/baseline.json`, `reports/finetuned.json` | Historical synthetic evaluation predictions; retain local evidence, exclude generated outputs from delivery. |
| `colab_training_bundle.zip` | Regenerable output of `scripts/prepare_colab_dataset.py`; retain source script and local archive. |
| Root `training_log.json` | Legacy console/history artifact outside all selected successful runs; preserve locally, exclude from delivery. |
| `data/invoices/*.jpg` | Invoice 01 visibly contains patient name, UHID, age/sex and doctor fields. Fictional status/publication rights are unverified. Preserve all five original images locally and all non-sensitive line-item labels; exclude images from delivery pending clearance. |
| `adapter/`, root `run_config.json`, root `training_result.json` | Legacy Colab snapshot, not selected by any final pointer; future notebook/archive use is uncertain. Proposed further exclusion only after owner review; currently retained/tracked. |

CORD source ZIPs, selected large receipt dataset, database files, caches, virtual
environments, `.env`/credentials, failed/smoke/checkpoint directories and local
reports remain on disk and ignored. No uncertain source module is classified as
junk. No data or model was deleted. Original fixtures remain usable locally.

**Publication blocker:** originals already exist in Git history. Index-only
exclusion prevents inclusion in the new tree but cannot erase past commits.
Before pushing a public deliverable, confirm they are synthetic/authorized or
obtain a separately approved history/publication plan. Do not rewrite history or
force-push during this task. Review existing tracked assets and selected metadata
for confidentiality before staging; no comprehensive privacy clearance is claimed.

## Manual delivery commands

Run from the repository root, on `repo_clean`, after reviewing this report and
the CPU/checksum commands in `docs/OPERATIONS.md`. No automatic commit/push.

First make Git LFS available manually using the VM's approved package-management
procedure. Then stop if any of these checks fails:

```bash
git branch --show-current
git status --short
git lfs version
git lfs install --local
git lfs env
git check-attr filter diff merge -- models/qwen25_vl_sft/runs/20261010T174916Z-0b733db4/adapter/adapter_model.safetensors models/qwen25_3b_reasoning_dpo/runs/20261011T074847Z-60ab777d/adapter/adapter_model.safetensors models/qwen25_vl_cord_sft/runs/20261011T134348Z-f1e33111/adapter/adapter_model.safetensors
```

Require filter=lfs, diff=lfs and merge=lfs for all three. Install does not use
`--force`: if an existing hook conflicts, preserve it and resolve deliberately.
Review the existing remote/LFS endpoint from `git lfs env` and account quota.

After approving the proposed generated/private exclusions, remove only their
index entries; `--cached` preserves the working copies:

```bash
git rm -r --cached -- reports/claim_graph reports/dpo reports/generated reports/inference
git rm --cached -- reports/baseline.json reports/finetuned.json colab_training_bundle.zip training_log.json
git rm -r --cached -- data/invoices
```

Do not run ordinary `git rm` or delete those directories. Preserve original
images externally for authorized reproducibility; image-based suites on a fresh
public clone need those local fixtures. The synthetic demo remains publishable
and supports browser/image demonstration without the originals.

Stage explicit review-prepared files and all three allowed final runs:

```bash
git add -- .gitignore .gitattributes README.md docs trainer evaluation notebooks/README.md models/README.md reports/README.md tests/README.md deploy/vllm/start_reasoning.sh
git add -- models/qwen25_vl_sft models/qwen25_3b_reasoning_dpo models/qwen25_vl_cord_sft
git diff --cached --check
git diff --cached --stat
git diff --cached --name-status
git lfs status
git lfs ls-files --long
```

Require exactly the three intended adapter weights in LFS. Check all other staged
paths: no full backbone, checkpoints, raw CORD, private images, credentials or
unreviewed outputs. Verify the staged objects are LFS pointer text, not binaries:

```bash
git show :models/qwen25_vl_sft/runs/20261010T174916Z-0b733db4/adapter/adapter_model.safetensors
git show :models/qwen25_3b_reasoning_dpo/runs/20261011T074847Z-60ab777d/adapter/adapter_model.safetensors
git show :models/qwen25_vl_cord_sft/runs/20261011T134348Z-f1e33111/adapter/adapter_model.safetensors
```

Each must show the LFS version line, expected SHA256 oid and size. Never print
these staged objects if `git lfs ls-files` did not first identify them as LFS.
Only after tests, hashes, privacy review and staged review are satisfactory:

```bash
git commit -m "Prepare director review docs, launchers and final LoRA artifacts"
git lfs fsck
git lfs ls-files --long
git lfs push --dry-run origin repo_clean
git status --short
git push -u origin repo_clean
```

Dry-run previews objects, not a guarantee of remote quota/authentication. The
normal LFS pre-push hook transfers objects; do not disable it. No `--force`,
history rewriting or automatic weight promotion. Do not push until the privacy
history blocker is cleared. Existing requirements, environment directories,
private source/report copies and temporary reasoning script are untouched.
