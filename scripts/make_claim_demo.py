"""Manually create a clearly synthetic 2024 image; never part of SFT data.

The reference CSV is used ONLY here to render the fixture. Pass ONLY the JPG
to scripts.run_claim_graph; extraction must come from invoice-extractor.
"""
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.adjudication.rules import parse_invoice_csv
from src.utils.paths import ROOT, project_path

DEMO_DATE = date(2024, 1, 15)
REFERENCE = ROOT / "data/evaluation/ground_truth/invoice_01_standard.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/demos/claim_2024"))
    args = parser.parse_args()
    directory = project_path(args.output_dir).resolve()
    if not directory.is_relative_to((ROOT / "data/demos").resolve()):
        parser.error("Demo output must be under data/demos, separate from training/validation")
    image_path = directory / "invoice_2024.jpg"
    metadata_path = directory / "demo.json"
    if image_path.exists() or metadata_path.exists():
        parser.error("Demo already exists; choose a new --output-dir (no overwrites)")
    invoice = parse_invoice_csv(REFERENCE, invoice_date=DEMO_DATE, invoice_id="SYNTHETIC-DEMO-2024")
    try:
        normal = ImageFont.truetype("DejaVuSans.ttf", 25)
        small = ImageFont.truetype("DejaVuSans.ttf", 20)
        heading = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
    except OSError:
        parser.error("DejaVuSans fonts unavailable; provide system fonts before running this demo generator")
    width, top, row_height = 1100, 225, 62
    height = top + row_height * (len(invoice.lines)+1) + 155
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 30), "DEMO HOSPITAL — SYNTHETIC INVOICE", font=heading, fill="black")
    draw.text((40, 85), "TEST FIXTURE ONLY — NO REAL PATIENT INFORMATION", font=small, fill="black")
    draw.text((40, 130), f"Invoice: {invoice.invoice_id}", font=normal, fill="black")
    draw.text((40, 172), f"Date: {DEMO_DATE.isoformat()}    Currency: INR", font=normal, fill="black")
    columns = [40, 535, 705, 875, 1060]
    bottom = top + row_height * (len(invoice.lines)+1)
    draw.rectangle((columns[0], top, columns[-1], top+row_height), fill="#e8edf3")
    for x in columns:
        draw.line((x, top, x, bottom), fill="black", width=2)
    for index in range(len(invoice.lines)+2):
        y = top + index * row_height
        draw.line((columns[0], y, columns[-1], y), fill="black", width=2)
    for x, label in zip(columns, ("Item", "Quantity", "Rate", "Amount")):
        draw.text((x+12, top+17), label, font=normal, fill="black")
    for index, line in enumerate(invoice.lines):
        values = (line.item, str(line.quantity), "" if line.rate is None else f"{line.rate:.2f}",
                  f"{line.amount:.2f}")
        for column, value in enumerate(values):
            if draw.textlength(value, font=normal) > columns[column+1]-columns[column]-24:
                parser.error("Reference value exceeds demo cell width; refusing to truncate invoice content")
            draw.text((columns[column]+12, top+(index+1)*row_height+17), value, font=normal, fill="black")
    draw.text((40, bottom+30), f"Total billed (INR): {invoice.billed_total:.2f}", font=normal, fill="black")
    draw.text((40, bottom+83), "Synthetic demo; policy eligibility is assessed separately.", font=small, fill="black")
    directory.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing fixtures even if another process runs.
    with image_path.open("xb") as stream:
        image.save(stream, format="JPEG", quality=95, subsampling=0)
    metadata = {"synthetic": True, "purpose": "image-only claim graph demonstration, never SFT",
        "invoice_date": DEMO_DATE.isoformat(), "image": str(image_path.relative_to(ROOT)),
        "source_csv": str(REFERENCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(), "rows": len(invoice.lines)}
    with metadata_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
