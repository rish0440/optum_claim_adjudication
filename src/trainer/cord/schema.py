"""Strict source normalization and shared, CPU-only dataset utilities."""
import csv
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import re

from src.inference.parsing import CSV_FIELDS
from src.prompts.extraction import INSTRUCTION
from src.schemas import InvoiceLine

REPO_ID = "Heliosoph/CORD"
DEFAULT_REVISION = "51b7e93"  # Explicit published commit prefix; resolved to full SHA on download.
SOURCE_COUNTS = {"train": 800, "validation": 100, "test": 100}
COUNTS = {"train": 161, "validation": 20, "test": 19}
TOTAL_COUNT = 200
FILES = tuple(name for split in COUNTS for name in
              (f"cord-{split}-images.zip", f"cord-{split}.jsonl.gz"))
MANIFESTS = {"train": "training.jsonl", "validation": "validation.jsonl", "test": "test.jsonl"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 20_000_000


def configured_counts(counts=None):
    """Validate requested split sizes; production always requires exactly 200."""
    counts = COUNTS if counts is None else counts
    if (not isinstance(counts, dict) or set(counts) != set(SOURCE_COUNTS)
            or any(type(value) is not int or value <= 0 for value in counts.values())):
        raise ValueError("Split counts must be positive integers for train, validation and test")
    if sum(counts.values()) != TOTAL_COUNT:
        raise ValueError(f"Split counts must total exactly {TOTAL_COUNT}")
    return {split: counts[split] for split in SOURCE_COUNTS}


class LabelError(ValueError):
    def __init__(self, reason, field, row_index=None):
        super().__init__(f"{reason}: {field}")
        self.reason, self.field, self.row_index = reason, field, row_index


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, values):
    with Path(path).open("x", encoding="utf-8") as stream:
        for value in values:
            stream.write(canonical(value) + "\n")


def read_jsonl(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    result = []
    with Path(path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record at line {index}")
            value = json.loads(line, object_pairs_hook=unique,
                               parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record must be an object at line {index}")
            result.append(value)
    return result


def quantity(value):
    if not isinstance(value, str):
        raise LabelError("invalid_quantity", "cnt")
    match = re.fullmatch(r"(?:[xX]\s*)?([1-9][0-9]*)", value.strip()) or re.fullmatch(
        r"([1-9][0-9]*)\s*[xX]", value.strip())
    if not match:
        raise LabelError("invalid_quantity", "cnt")
    return int(match.group(1))


def money(value, field):
    """Only explicit whole-rupiah labels: . and , are grouping separators.

    Decimal-looking/mixed notation is rejected, never guessed or converted.
    A single terminal period is accepted after otherwise valid grouping.
    """
    if not isinstance(value, str):
        raise LabelError("invalid_money", field)
    text = value.strip()
    if text.endswith("."):
        text = text[:-1]
    if re.fullmatch(r"[0-9]+", text):
        digits = text
    elif re.fullmatch(r"[0-9]{1,3}(?:,[0-9]{3})+", text):
        digits = text.replace(",", "")
    elif re.fullmatch(r"[0-9]{1,3}(?:\.[0-9]{3})+", text):
        digits = text.replace(".", "")
    else:
        raise LabelError("ambiguous_or_invalid_money", field)
    if len(digits) > 12:
        raise LabelError("money_out_of_range", field)
    return Decimal(digits).quantize(Decimal("0.01"))


def convert(record):
    """Reject the entire receipt if any menu row is incomplete; never drop rows."""
    try:
        menu = record["ground_truth"]["gt_parse"]["menu"]
    except (KeyError, TypeError):
        raise LabelError("missing_menu", "ground_truth.gt_parse.menu") from None
    menu = [menu] if isinstance(menu, dict) else menu
    if not isinstance(menu, list) or not menu:
        raise LabelError("invalid_menu", "menu")
    lines = []
    for index, row in enumerate(menu):
        try:
            if not isinstance(row, dict):
                raise LabelError("invalid_menu_row", "menu")
            for field in ("nm", "cnt", "unitprice", "price"):
                if field not in row or row[field] is None or row[field] == "":
                    raise LabelError("missing_field", field)
            if any(isinstance(value, (dict, list)) for value in row.values()):
                raise LabelError("unsupported_nested_menu", "menu")
            item = row["nm"]
            if not isinstance(item, str) or not item.strip() or any(ord(c) < 32 and c not in "\n\r\t" for c in item):
                raise LabelError("invalid_item_name", "nm")
            count, rate, amount = quantity(row["cnt"]), money(row["unitprice"], "unitprice"), money(row["price"], "price")
            if Decimal(count) * rate != amount:
                raise LabelError("arithmetic_mismatch", "cnt/unitprice/price")
            lines.append(InvoiceLine(item=item, quantity=count, rate=rate, amount=amount))
        except LabelError as exc:
            exc.row_index = index
            raise
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(CSV_FIELDS)
    writer.writerows((line.item, line.quantity, f"{line.rate:.2f}", f"{line.amount:.2f}") for line in lines)
    return stream.getvalue(), lines


def messages(image, csv_text):
    """The exact corrected multimodal representation used by make_hf_dataset."""
    return {"images": [image], "prompt": [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": INSTRUCTION}]}],
        "completion": [{"role": "assistant", "content": [{"type": "text", "text": csv_text}]}]}
