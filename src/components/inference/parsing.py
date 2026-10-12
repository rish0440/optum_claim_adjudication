"""Shared CSV parsing, preserving the original extraction/scoring semantics."""
import csv
import io

CSV_FIELDS = ("Item", "Quantity", "Rate", "Amount")


def read_rows(text: str) -> list[dict[str, str]]:
    """Parse strict four-column CSV, retaining intentionally empty Rate cells."""
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff").strip()))
    if reader.fieldnames is None or tuple(reader.fieldnames) != CSV_FIELDS:
        raise ValueError(f"CSV must start with {','.join(CSV_FIELDS)}")
    rows = []
    for line in reader:
        if None in line or any(key not in line for key in CSV_FIELDS):
            raise ValueError("CSV row has extra/missing columns")
        if not line["Item"].strip() or not line["Quantity"].strip() or not line["Amount"].strip():
            raise ValueError("Missing Item, Quantity or Amount")
        rows.append({k: line[k].strip() for k in CSV_FIELDS})
    if not rows:
        raise ValueError("CSV has no invoice items")
    return rows


def clean_csv_text(value: str) -> str:
    """Strip Markdown fences; never manufacture missing predictions."""
    text = value.strip().removeprefix("\ufeff")
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
    return text + ("\n" if not text.endswith("\n") else "")


def get_predicted_rows(text: str) -> list[dict[str, str]]:
    """Require an exact CSV header. Bad generations score as failures."""
    return read_rows(clean_csv_text(text))
