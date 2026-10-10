# Assignment data

- `policy.json`: unmodified source policy, authoritative for exact coverage and exclusion rule IDs.
- `invoices/`: five original supplied JPGs, **held out from model fine-tuning**.
- `train/`: future synthetic invoice-image + label pairs; generated separately.
- `evaluation/`: future independently checked expected line items and decisions.

Do not train on, augment into training, or otherwise leak the five original evaluation invoices.
