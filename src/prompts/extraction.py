"""Original image-to-CSV instruction shared by training and inference."""
INSTRUCTION = (
    "Extract every line item visible in this hospital invoice. "
    "Return ONLY CSV text with the exact header Item,Quantity,Rate,Amount. "
    "Preserve row order and item names. Copy numeric amounts without currency "
    "symbols or thousands separators. If the rate column is not printed, "
    "leave Rate empty; do not calculate or guess missing rates. "
    "Do not include totals, explanations, or Markdown fences."
)
