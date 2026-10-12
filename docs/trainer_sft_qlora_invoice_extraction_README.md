# Phase 10A — strict CORD 200 preparation

Code preparation only. No download, test, training, model loading or installation
was performed by Codex. Nothing in the working application or existing adapters
is changed. All output counts and receipt eligibility await manual execution.

## Source and attribution

[Heliosoph/CORD dataset README](https://huggingface.co/datasets/Heliosoph/CORD)
is the source contract: train 800, validation 100, test 100; separate JPEG ZIP
and gzipped annotation JSONL per split; image_file joins directly to a ZIP member.
The mirror attributes original CORD to NAVER Clova and Park et al., CC BY 4.0.
Images are upstream receipts re-encoded to JPEG; structured annotations are
preserved. This is Indonesian restaurant/retail extraction data, not insurance
policy reasoning or an INR dataset.

Only these six files are fetched, using individual Hub downloads:

```
cord-train-images.zip           cord-train.jsonl.gz
cord-validation-images.zip      cord-validation.jsonl.gz
cord-test-images.zip            cord-test.jsonl.gz
```

Default revision is the published commit prefix `51b7e93`, not moving `main`.
The downloader resolves it to a full SHA, uses that SHA for every file, and
records Hub file sizes/SHA256/URLs in source_manifest.json. Supply a full SHA
to --revision if desired. It does not fetch a repository snapshot or models.

The existing huggingface_hub source supports dataset_info(files_metadata=True),
hf_hub_download(local_dir=...) and automatic partial-download resumption. Native
progress bars and three bounded file retries are enabled. Existing files are
hashed before reuse; corrupt files are not silently overwritten or deleted.
Use another source directory to recover a corrupt final file. The source
directory cannot be reused for a different revision.

## Strict eligibility and selection

- Every menu row must have nonempty nm, cnt, **explicit** unitprice and price.
  menu can be one dictionary or a list. Original row order, repeated items and
  full original ground_truth are retained. A bad row rejects the entire receipt,
  avoiding incomplete extraction targets. Nested menu sub-items are rejected
  rather than silently discarded.
- Positive integer quantities accept `x 2`, `2`, `2X` and equivalent whitespace.
- Whole-rupiah grouping accepts `75,000`, `75.000`, `1.234.567`, `40,000.`.
  Plain digits are accepted; mixed separators, decimal-looking values, currency
  prefixes, negative or invalid labels are rejected. This intentionally excludes
  uncertain formats; a period is never automatically treated as a decimal.
- Values use Decimal, serialize with two decimal places, and must satisfy
  quantity × explicit rate = line amount exactly. Explicit zero-rate/zero-amount
  items are retained. No derived rates, rounding tolerances or currency conversion.
- Only gt_parse.menu is converted. Receipt-level total, subtotal, tax, gratuity,
  payments and discounts outside menu are not line items. No missing rows are
  guessed, generated or borrowed from the Optum/synthetic datasets.
- JPEG decoding, annotation dimensions, maximum 8 MiB / 20 MP, unique source
  filenames, image-byte hashes and decoded RGB pixel hashes are checked.
- Seed 42, source-order-independent sorting and stratified seeded selection
  prefer multi-row receipts; size, aspect ratio, row count and name length are
  diversity proxies, not claims of human-verified layout quality.
- Approved default selection is **161 train / 20 validation / 19 test** (200 total).
  The counts are configurable through --train-count, --validation-count and
  --test-count; all three must be positive integers totaling exactly 200. Each example
  stays in its original source split. If any split lacks enough strict receipts,
  preparation stops, writes eligibility/rejection reports and produces no dataset.
  Labels are never relaxed merely to reach 200. Configured counts are recorded
  in the manifest and reports; validation enforces those counts and the 200 total.

## Actual trainer contract and compatibility boundary

The existing raw SFT manifest uses image, csv, target_csv and rows, not a plain
text-only prompt/completion dataset. The shared corrected make_hf_dataset builds:

```json
{
  "images": ["train/images/cord-train-0000.jpg"],
  "prompt": [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "<unchanged src.components.prompts.extraction.INSTRUCTION>"}
  ]}],
  "completion": [{"role": "assistant", "content": [
    {"type": "text", "text": "Item,Quantity,Rate,Amount\n...\n"}
  ]}]
}
```

Each output record contains those messages **and** the existing raw manifest
fields, plus source_filename/source_id/source_split/provenance. Paths are
relative to the selected dataset root; validation resolves them to existing
local files before passing records to the original HF builder. Image content
is carried in its datasets.Image feature, not embedded as a string in messages.
There is no model/tokenizer download in --check-hf.

The exact existing production extraction instruction is reused, including its
hospital wording. No new competing prompt is introduced. Its wording is a
domain limitation to consider explicitly in Phase 10B.

**Current training CLI is not yet CORD-compatible:** read_records requires a
root named synthetic; verify_dataset additionally requires synthetic invoice
IDs/layout metadata and excludes labels resembling Optum reference targets.
Phase 10A deliberately does not modify either guard or trainer. The shared HF
builder/CSV schema is validated here; direct training with train_qlora still
requires a focused Phase 10B dataset-preflight adaptation. Do not rename CORD
to synthetic, bypass safety guards, or train before approving that adaptation.

## Manual commands from the VM repository root

```bash
cd /home/rishabh_mishra0440/optum_claim_adjudication/optum_claim_adjudication
source ../.venv/bin/activate

# A: CPU-only unit fixtures; no remote dataset or GPU access
python -B -m unittest tests.test_cord_training -v

# B: Reuse the already downloaded and verified files in .cache/cord_source.
# Do not download them again for this correction.

# C: Eligibility only; decode candidates in memory, save no receipt images
python -u -m trainer.sft_qlora_invoice_extraction.data_prep prepare --source-dir .cache/cord_source --seed 42 --train-count 161 --validation-count 20 --test-count 19 --preflight

# D: Exactly 161/20/19; existing output directories are never overwritten
python -u -m trainer.sft_qlora_invoice_extraction.data_prep prepare --source-dir .cache/cord_source --output-dir data/cord_sft_qlora --seed 42 --train-count 161 --validation-count 20 --test-count 19

# E: Validate original compressed annotation references and shared HF builder
python -u -m trainer.sft_qlora_invoice_extraction.data_prep validate --dataset-root data/cord_sft_qlora --source-dir .cache/cord_source --check-hf

# F: Print actual paired image paths and first two CSV targets per split
python -u -m trainer.sft_qlora_invoice_extraction.data_prep validate --dataset-root data/cord_sft_qlora --examples 2

# Optional: first four selected receipts per split; use a fresh PNG name
python -u -m trainer.sft_qlora_invoice_extraction.data_prep validate --dataset-root data/cord_sft_qlora --contact-sheet reports/cord/contact-sheet-200.png

# G: OPTIONAL explicit deletion of only the three verified temporary ZIPs
python -u -m trainer.sft_qlora_invoice_extraction.data_prep download --source-dir .cache/cord_source --dataset-root data/cord_sft_qlora --cleanup-archives
```

Preflight reports are saved under a unique reports/cord/eligibility-<run-id>/
directory. --report-dir may select a fresh directory. Failures report filename,
source split/line, row index, reason and missing/invalid field; no source record
is omitted from the eligible/rejected accounting.

## Prepared output (created only by manual execution)

```
.cache/cord_source/                 # six archives, source_manifest.json, local Hub metadata
data/cord_sft_qlora/
  train/
    images/                      # 161 original JPEG filenames
    labels/                      # 161 corresponding CSV files
    annotations.jsonl            # 161 untouched structured source records
    training.jsonl               # 161 raw + multimodal SFT records
  validation/
    images/                      # 20
    labels/                      # 20
    annotations.jsonl
    validation.jsonl
  test/
    images/                      # 19, untouched by future training/tuning
    labels/                      # 19
    annotations.jsonl
    test.jsonl
  manifest.json                  # pinned source, pair fingerprints, file fingerprints
  selection_report.json          # eligibility, statistics, examples, limitations
  rejection_report.json          # every rejected source record and its reason
  validation_report.json         # initial successful CPU validation
  README.md
```

Preparation performs full local validation before writing validation_report.json.
Any interrupted preparation leaves its partial files for inspection, with no
success report; choose a new output directory to retry. Standalone validation
recomputes facts/hashes rather than trusting an old success report. Supplying
--source-dir verifies raw selected annotations against their original gzipped
JSONL rows; it still works after deleting the ZIPs. --check-hf validates the
three splits' corrected image/chat representation, not a forward pass or trainer
CLI success. Human visual review remains pending even after automated PASS.

## Storage, dependencies and cleanup

The Hub currently lists about 212 MB for all files; selected images add their
actual sizes (not all 1,000 images). Reserve approximately 2 GB free as a
conservative starting point. Runtime checks use actual Hub sizes and free disk
space before downloading/preparation. Eligibility reads ZIP members in memory;
only the 200 selected images are written. No archive.extractall is used.

Uses existing Pillow/Pydantic plus stdlib, existing huggingface_hub for download,
and installed datasets only for --check-hf. No requirements changes or installs.
If a package is missing, stop and report it; do not upgrade the working stack.

Cleanup validates both all source hashes and the prepared dataset, then removes
only the three named ZIPs. It preserves gzip annotations, manifests, images,
labels and all caches. Source ZIPs must be re-downloaded before selecting again.
No other Hugging Face caches or user files are deleted. Default source/output/
review paths have specific .gitignore entries; custom paths need equivalent
Git safety before use. Do not commit downloaded receipts or archives.

The five Optum invoices, existing synthetic dataset, model weights, reasoning
agents, Milvus, policy, LangGraph, application/UI/PDF and financial rules are
untouched. Phase 10B training and adapter evaluation are outside this module.
