# Five Optum invoices — evaluation/test ONLY

`data/invoices/` is the original untouched supplied JPG evaluation set.
`data/evaluation/ground_truth/` contains line-by-line manually transcribed CSV
and matching JSON, plus an initial policy-based expected-adjudication JSON.

The annotation creation script is `scripts/make_optum_ground_truth.py`. It
contains explicit human-reviewed source values, not an OCR/VLM predictor.
The five images and their labels are **never used to train, fine-tune or tune**
the VLM. The synthetic validation split is distinct from Optum's final test set.

Rate values are blank in invoice 04 because that column is **not shown**.
Surgical Gloves in invoice 05 are mapped provisionally to COV-019, which must
be documented as an assumption when evaluating adjudication.

The expected-adjudication file is a reference for later model evaluation,
NOT actual adjudication-engine output or evidence of model accuracy.
