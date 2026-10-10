"""Generate paired synthetic hospital-invoice JPG images and exact CSV labels.

Usage from repository root:
    python -m src.trainer.sft.annotation.generator --count 10 --seed 42
    python -m src.trainer.sft.annotation.generator --verify
    python -m src.trainer.sft.annotation.generator --count 10 --seed 42 --overwrite

Nothing here opens or copies the five Optum held-out evaluation invoices.
CORD receipt files are also intentionally excluded from training-data generation.
"""

import argparse
import csv
import io
import json
import random
import shutil
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, __version__ as PILLOW_VERSION

from src.schemas import InvoiceLine
from src.config.settings import get_settings
from .validation import verify_dataset

CSV_COLUMNS = ("Item", "Quantity", "Rate", "Amount")
STYLES = ("bordered", "semi_bordered", "borderless")
PALETTES = (
    ("#125781", "#eaf4fb", "#315477"),
    ("#148266", "#ebf9f3", "#286a57"),
    ("#934c66", "#fff0f4", "#713a50"),
    ("#6c5f9b", "#f4f1ff", "#51457b"),
)
HOSPITALS = (
    "NORTHGATE MEDICAL CENTRE", "GREEN VALLEY HOSPITAL", "RIVERBEND HEALTHCARE",
    "SUNRISE CLINICAL CENTRE", "OAKRIDGE HOSPITAL", "CLEARWATER DIAGNOSTICS",
    "SUMMIT MULTISPECIALITY", "BLUEBIRD MEDICAL CENTRE",
)
# Medical invoice categories deliberately resemble real healthcare charges, but
# generated line values/layouts do not reproduce the held-out Optum invoices.
ITEM_RATES = {
    "Room Charges": (1800, 5200),
    "ICU Charges": (4500, 10000),
    "Consultation": (450, 2100),
    "Specialist Consultation": (850, 3200),
    "Medicines": (75, 1200),
    "Blood Test (CBC)": (250, 1100),
    "Blood Test (Lipid Profile)": (500, 1750),
    "CT Scan": (3000, 7500),
    "MRI Scan": (6000, 13500),
    "X-Ray": (450, 1850),
    "Ultrasound": (750, 3650),
    "ECG": (300, 1300),
    "Physiotherapy": (450, 1700),
    "Nursing Charges": (600, 1900),
    "Ambulance": (800, 2900),
    "Surgical Gloves": (140, 580),
    "Oxygen Charges": (350, 2900),
    "Medical Consumables": (180, 1900),
    "Diet Meal": (140, 650),
    "Television Charges": (150, 550),
    "Personal Care Kit": (280, 850),
    "Health Supplement": (200, 800),
}
FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
)
BOLD_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
)


def get_font(size: int, *, bold: bool = False, choice: int = 0) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Use an installed font; fall back to Pillow's built-in font on any OS."""
    options = BOLD_FONTS if bold else FONTS
    available = [name for name in options if Path(name).is_file()]
    if available:
        return ImageFont.truetype(available[choice % len(available)], size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Older Pillow versions do not accept the size argument.
        return ImageFont.load_default()


def draw_right(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: str) -> None:
    """Place numeric text right-aligned at x."""
    draw.text(xy, text, font=font, fill=fill, anchor="ra")


def make_lines(rng: random.Random, min_rows: int, max_rows: int, omit_rate: bool) -> list[InvoiceLine]:
    """Generate actual structured data BEFORE rendering an invoice image."""
    count = rng.randint(min_rows, min(max_rows, len(ITEM_RATES)))
    names = rng.sample(list(ITEM_RATES), count)
    lines: list[InvoiceLine] = []
    for name in names:
        lo, hi = ITEM_RATES[name]
        # Multiples of ten are common but deliberately allow varied quantities.
        rate = rng.randrange((lo + 9) // 10, hi // 10 + 1) * 10
        if name in {"Room Charges", "ICU Charges", "Consultation", "Specialist Consultation", "MRI Scan", "CT Scan", "X-Ray"}:
            qty = rng.randint(1, 3)
        else:
            qty = rng.randint(1, 8)
        amount = rate * qty
        lines.append(InvoiceLine(item=name, quantity=qty, rate=None if omit_rate else Decimal(rate), amount=Decimal(amount)))
    return lines


def lines_as_csv(lines: list[InvoiceLine]) -> str:
    """Serialize exact image-visible field values to a normalized CSV string."""
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for line in lines:
        writer.writerow((line.item, line.quantity, "" if line.rate is None else str(line.rate), str(line.amount)))
    return stream.getvalue()


def _fit_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, size: int, font_choice: int, *, bold: bool = False) -> ImageFont.ImageFont:
    """Reduce a long service name's font size so it never overlaps quantity."""
    for s in range(size, 13, -1):
        font = get_font(s, choice=font_choice, bold=bold)
        if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
            return font
    return get_font(13, choice=font_choice, bold=bold)


def draw_table_cell(draw, bounds, text, font, fill, *, right_aligned=False):
    """Center text vertically and reject clipping before an invoice is saved."""
    left, top, right, bottom = bounds
    anchor = "rm" if right_aligned else "lm"
    xy = (right - 15 if right_aligned else left + 13, (top + bottom) / 2)
    box = draw.textbbox(xy, text, font=font, anchor=anchor)
    if not (left < box[0] and box[2] < right and top < box[1] and box[3] < bottom):
        raise ValueError(f"Invoice text does not fit its table cell: {text!r}")
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def render_invoice(
    image_path: Path, lines: list[InvoiceLine], *, rng: random.Random,
    style: str, font_choice: int, palette_index: int, invoice_number: int,
) -> dict:
    """Render a healthcare-looking JPG from the EXACT SAME list as the CSV."""
    assert style in STYLES
    has_rate = all(line.rate is not None for line in lines)
    width = rng.choice((900, 980, 1040, 1080))
    row_height = rng.choice((43, 48, 53))
    top = 295 + rng.randint(-8, 12)
    header_height = 49
    bottom = top + header_height + row_height * len(lines)
    height = bottom + 185
    image = Image.new("RGB", (width, height), "#fffefd")
    draw = ImageDraw.Draw(image)
    brand, pale, deep = PALETTES[palette_index % len(PALETTES)]
    regular = get_font(19, choice=font_choice)
    small = get_font(15, choice=font_choice)
    label_font = get_font(18, bold=True, choice=font_choice)
    heading_font = get_font(27, bold=True, choice=font_choice)
    number_font = get_font(19, choice=font_choice)
    left, right = 65, width - 65

    # Letterhead / invoice metadata. Fictional hospital and patient identifiers.
    draw.rounded_rectangle((left, 45, left + 58, 103), radius=10, fill=brand)
    draw.rectangle((left + 24, 57, left + 34, 92), fill="white")
    draw.rectangle((left + 12, 69, left + 46, 80), fill="white")
    hospital = HOSPITALS[invoice_number % len(HOSPITALS)]
    draw.text((left + 76, 42), hospital, font=heading_font, fill=deep)
    draw.text((left + 77, 83), "Central Avenue, India  |  Tel: 080-00000000", font=small, fill="#596574")
    draw.line((left, 126, right, 126), fill=brand, width=3)
    draw.text((left, 148), rng.choice(("HOSPITAL INVOICE", "ITEMIZED BILL", "PATIENT INVOICE", "BILL / INVOICE")), font=heading_font, fill="#26333e")
    docdate = date(2025, 1, 1) + timedelta(days=rng.randint(0, 320))
    draw.text((left, 191), f"Invoice No: SYN/{2025}/{invoice_number:05d}", font=regular, fill="#26333e")
    draw.text((left, 223), f"Date: {docdate.strftime('%d %b %Y')}", font=small, fill="#546373")
    draw_right(draw, (right, 195), f"Patient: Sample-{invoice_number:05d}", regular, "#26333e")
    draw_right(draw, (right, 223), "Currency: INR", small, "#546373")

    # Table column boundaries vary when the unit-rate column is not printed.
    if has_rate:
        positions = [left, left + round((right-left)*0.51), left + round((right-left)*0.65), left + round((right-left)*0.83), right]
        headings = ("Service / Item", "Qty", "Rate (INR)", "Amount (INR)")
    else:
        positions = [left, left + round((right-left)*0.61), left + round((right-left)*0.76), right]
        headings = ("Service / Item", "Qty", "Amount (INR)")

    draw.rectangle((left, top, right, top+header_height), fill=pale)
    if style == "bordered":
        for i in range(len(lines)):
            if i % 2:
                y0 = top + header_height + i * row_height
                draw.rectangle((left + 2, y0 + 1, right - 2, y0 + row_height - 1), fill="#f8fafb")
    if style == "bordered":
        draw.rectangle((left, top, right, bottom), outline="#8c9fac", width=2)
        for x in positions[1:-1]:
            draw.line((x, top, x, bottom), fill="#c8d4dc", width=2)
        draw.line((left, top + header_height, right, top + header_height), fill="#8c9fac", width=2)
    elif style == "semi_bordered":
        draw.line((left, top, right, top), fill="#a1b2bd", width=2)
        draw.line((left, top + header_height, right, top + header_height), fill="#98adb7", width=2)
    else:
        draw.line((left, top + header_height, right, top + header_height), fill="#abb6be", width=2)

    for c, heading in enumerate(headings):
        font = _fit_text(draw, heading, positions[c+1]-positions[c]-30, 18 if c == 0 else 16,
                         font_choice, bold=True)
        draw_table_cell(draw, (positions[c], top, positions[c+1], top+header_height),
                        heading, font, deep, right_aligned=c > 0)

    for i,line in enumerate(lines):
        y0 = top + header_height + i*row_height
        y1 = y0 + row_height
        if style in {"bordered", "semi_bordered"}:
            draw.line((left,y1,right,y1),fill="#d9e1e6",width=1)
        item_font = _fit_text(draw,line.item,positions[1]-positions[0]-27,19,font_choice)
        draw_table_cell(draw, (positions[0], y0, positions[1], y1), line.item, item_font, "#25323d")
        entries = [str(line.quantity)]
        if has_rate:
            entries.append(f"{int(line.rate):,}")
        entries.append(f"{int(line.amount):,}")
        for c,value in enumerate(entries, start=1):
            draw_table_cell(draw, (positions[c], y0, positions[c+1], y1),
                            value, number_font, "#25323d", right_aligned=True)

    total = sum(int(line.amount) for line in lines)
    draw.line((left,bottom+30,right,bottom+30),fill="#d8e1e8",width=2)
    draw.text((right-315,bottom+48),"GRAND TOTAL (INR)",font=label_font,fill=deep)
    draw_right(draw,(right-8,bottom+73),f"{total:,}",heading_font,deep)
    draw.text((left,bottom+130),"This is a computer-generated invoice  |  Billing Department",font=small,fill="#697783")
    image_path.parent.mkdir(parents=True,exist_ok=True)
    image.save(image_path,format="JPEG",quality=93,subsampling=0)
    return {"invoice_id": image_path.stem, "width": width, "height": height,
            "row_height": row_height, "table_top": top,
            "font_family": regular.getname()[0], "font_path": str(getattr(regular, "path", "Pillow default"))}


def _paths_for(name: str, split: str, base: Path) -> tuple[Path,Path]:
    return (base / split / "images" / f"{name}.jpg", base / split / "labels" / f"{name}.csv")


def generate_dataset(
    count: int = 10, seed: int = 42, train_ratio: float = 0.8,
    min_rows: int = 3, max_rows: int = 9, missing_rate_fraction: float = 0.2,
    overwrite: bool = False,
) -> dict:
    """Generate train/validation images and csv with a reproducible random seed."""
    if count < 2:
        raise ValueError("count must be >= 2 to create both train and validation")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if min_rows < 1 or min_rows > max_rows or max_rows > len(ITEM_RATES):
        raise ValueError("Use 1 <= min_rows <= max_rows <= number of item types")
    if not 0 <= missing_rate_fraction <= 1:
        raise ValueError("missing_rate_fraction must be between 0 and 1")
    settings = get_settings()
    base = settings.paths.synthetic_dataset.resolve()
    if base.name != "synthetic":
        raise ValueError("Dataset root must be a directory named 'synthetic'")
    # Reject overlap in either direction before any overwrite, including symlinks.
    for protected in (settings.paths.optum_invoices, settings.paths.evaluation_ground_truth, settings.paths.cord):
        protected = protected.resolve()
        if base.is_relative_to(protected) or protected.is_relative_to(base):
            raise ValueError("Synthetic output must not overlap held-out or third-party data")
    for split in ("train", "validation"):
        if (base / split).is_symlink():
            raise ValueError("Synthetic split directories must not be symlinks")
    if (base / "dataset_info.json").exists() or any((base / x).exists() for x in ("train", "validation")):
        if not overwrite:
            raise FileExistsError("Synthetic dataset exists. Pass --overwrite to regenerate it.")
        for split in ("train", "validation"):
            shutil.rmtree(base / split, ignore_errors=True)
    base.mkdir(parents=True,exist_ok=True)
    rng = random.Random(seed)
    train_count = min(count-1, max(1, round(count * train_ratio)))
    assigned = ["train"]*train_count + ["validation"]*(count-train_count)
    rng.shuffle(assigned)
    manifest_by_split: dict[str,list[dict]] = {"train":[],"validation":[]}
    layout_counts: Counter[str] = Counter()
    missing_rate_count=0
    rendering = []
    # Include a missing-rate-column example in BOTH splits when possible so
    # this edge case can be trained on and validated independently.
    missing_count = round(count * missing_rate_fraction)
    missing_indices: set[int] = set()
    selector = random.Random(seed + 999)  # Do not perturb invoice content RNG.
    by_split = {
        split: [i for i, value in enumerate(assigned, 1) if value == split]
        for split in ("train", "validation")
    }
    if missing_count >= 2:
        missing_indices.add(selector.choice(by_split["train"]))
        missing_indices.add(selector.choice(by_split["validation"]))
    remaining = [i for i in range(1, count + 1) if i not in missing_indices]
    missing_indices.update(selector.sample(remaining, missing_count - len(missing_indices)))
    for idx in range(1,count+1):
        split=assigned[idx-1]
        name=f"invoice_{idx:05d}"
        style=STYLES[(idx-1)%len(STYLES)]
        missing_rate = idx in missing_indices
        lines=make_lines(rng,min_rows,max_rows,omit_rate=missing_rate)
        img_path,csv_path = _paths_for(name,split,base)
        rendering.append(render_invoice(img_path,lines,rng=rng,style=style,font_choice=idx%3,palette_index=(idx-1)%len(PALETTES),invoice_number=idx))
        csv_path.parent.mkdir(parents=True,exist_ok=True)
        label=lines_as_csv(lines)
        csv_path.write_text(label,encoding="utf-8",newline="")
        record={
            "image": str(img_path.relative_to(base)).replace('\\','/'),
            "csv": str(csv_path.relative_to(base)).replace('\\','/'),
            "target_csv": label,
            "layout": style,
            "rate_column_visible": not missing_rate,
            "rows":len(lines),
            "billed_total":str(sum((x.amount for x in lines),Decimal(0))),
        }
        manifest_by_split[split].append(record)
        layout_counts[style]+=1
        missing_rate_count+=int(missing_rate)
    for split, records in manifest_by_split.items():
        with (base / split / "annotations.jsonl").open("w",encoding="utf-8") as out:
            for record in records:
                out.write(json.dumps(record,ensure_ascii=False)+"\n")
    info={"source":"synthetic_hospital_invoice_generator", "seed":seed,"total":count,
          "train":train_count,"validation":count-train_count,"layouts":dict(layout_counts),
          "missing_rate_column_images":missing_rate_count,
          "csv_columns":list(CSV_COLUMNS),"eval_set_excluded":True,
          "cord_included":False,
          "pillow_version": PILLOW_VERSION, "rendering": rendering}
    (base / "dataset_info.json").write_text(json.dumps(info,indent=2)+"\n",encoding="utf-8")
    return verify_dataset(base)




def main() -> None:
    parser=argparse.ArgumentParser(description="Generate synthetic hospital invoice images + exact CSV labels")
    parser.add_argument("--count",type=int,default=None)
    parser.add_argument("--seed",type=int,default=None)
    parser.add_argument("--train-ratio",type=float,default=None)
    parser.add_argument("--min-rows",type=int,default=None)
    parser.add_argument("--max-rows",type=int,default=None)
    parser.add_argument("--missing-rate-fraction",type=float,default=None)
    parser.add_argument("--overwrite",action="store_true",help="Replace only existing synthetic train/validation output")
    parser.add_argument("--verify",action="store_true",help="Verify a previously generated dataset")
    args=parser.parse_args()
    if args.verify:
        summary=verify_dataset()
    else:
        conf=get_settings().synthetic
        summary=generate_dataset(
            count=args.count if args.count is not None else conf.initial_count,
            seed=args.seed if args.seed is not None else conf.seed,
            train_ratio=args.train_ratio if args.train_ratio is not None else conf.train_ratio,
            min_rows=args.min_rows if args.min_rows is not None else conf.min_rows,
            max_rows=args.max_rows if args.max_rows is not None else conf.max_rows,
            missing_rate_fraction=args.missing_rate_fraction if args.missing_rate_fraction is not None else conf.missing_rate_fraction,
            overwrite=args.overwrite,
        )
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
