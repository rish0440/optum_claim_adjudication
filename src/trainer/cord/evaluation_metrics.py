"""Strict CSV scoring with Decimal values and deterministic one-to-one alignment."""
import csv
from decimal import Decimal, InvalidOperation
import io
import re

from src.inference.parsing import CSV_FIELDS, read_rows


def normalize(row, *, allow_empty_rates=False):
    result = [" ".join(row["Item"].split())]
    for field in CSV_FIELDS[1:]:
        value = row[field].strip()
        if field == "Rate" and not value and allow_empty_rates:
            result.append("")  # A visibly absent rate is a value, never a derived price.
            continue
        try:
            if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value):
                raise ValueError("not a plain nonnegative decimal")
            number = Decimal(value)
            if field == "Quantity" and (number <= 0 or number != number.to_integral_value()):
                raise ValueError("quantity must be a positive integer")
        except (ValueError, InvalidOperation):
            number = None
        result.append(number)
    return tuple(result)


def parse_prediction(raw):
    if not isinstance(raw, str) or not raw.strip() or "```" in raw:
        raise ValueError("Empty output or forbidden Markdown")
    table = list(csv.reader(io.StringIO(raw.strip()), strict=True))
    if not table or tuple(table[0]) != CSV_FIELDS:
        raise ValueError("Exact Item,Quantity,Rate,Amount header required")
    if len(table) < 2 or any(len(row) != 4 for row in table[1:]):
        raise ValueError("CSV must contain nonempty four-column item rows only")
    # Reuse the runtime structural parser without its Markdown-cleanup helper.
    return read_rows(raw)


def align_rows(expected, predicted):
    """Greedy best agreement; tie break by reference then prediction position.

    Eligible pairs share the whitespace-normalized item name OR all three valid
    numeric fields. Exact four-field rows win first. Each row is used at most once.
    No fuzzy/semantic name rewriting, rate derivation or positional assumption.
    """
    candidates = []
    for i, truth in enumerate(expected):
        for j, actual in enumerate(predicted):
            matches = [a is not None and a == b for a, b in zip(truth, actual)]
            if matches[0] or all(matches[1:]):
                candidates.append((-sum(matches), -int(matches[0]), i, j))
    used_expected, used_predicted, matches = set(), set(), []
    for _, _, i, j in sorted(candidates):
        if i not in used_expected and j not in used_predicted:
            matches.append((i, j))
            used_expected.add(i)
            used_predicted.add(j)
    return sorted(matches), sorted(set(range(len(expected))) - used_expected), sorted(
        set(range(len(predicted))) - used_predicted)


def score(expected_csv, prediction, *, allow_empty_rates=False):
    assessed = expected_csv is not None
    truth_rows = parse_prediction(expected_csv) if assessed else []
    expected = [normalize(row, allow_empty_rates=allow_empty_rates) for row in truth_rows]
    if any(value is None for row in expected for value in row):
        raise ValueError("Invalid ground-truth numeric fields")
    raw, error = prediction.get("raw_response", ""), prediction.get("error")
    parsed, parse_error = [], None
    try:
        parsed = parse_prediction(raw)
    except (ValueError, csv.Error) as exc:
        parse_error = str(exc)
    normalized = [normalize(row, allow_empty_rates=allow_empty_rates) for row in parsed]
    invalid = [{"prediction_index": i, "field": field} for i, row in enumerate(normalized)
               for field, value in zip(CSV_FIELDS[1:], row[1:]) if value is None]
    format_valid = parse_error is None and not invalid
    eligible = format_valid and not error and not prediction.get("truncated", False)
    # Incomplete generations/errors score as failures, even if their prefix is CSV.
    scoring_rows = normalized if not error and not prediction.get("truncated", False) else []
    matched, missing, extra = align_rows(expected, scoring_rows)
    field_correct = {field: 0 for field in CSV_FIELDS}
    details, complete = [], 0
    for i, j in matched:
        mismatches = [field for field, a, b in zip(CSV_FIELDS, expected[i], scoring_rows[j])
                      if b is None or a != b]
        for field in CSV_FIELDS:
            field_correct[field] += int(field not in mismatches)
        complete += int(not mismatches)
        details.append({"reference_index": i, "prediction_index": j, "mismatched_fields": mismatches})
    arithmetic = []
    for i, row in enumerate(normalized):
        q, rate, amount = row[1:]
        if None not in (q, rate, amount) and rate != "" and q * rate != amount:
            arithmetic.append(i)
    return {"csv_valid": format_valid, "eligible_completion": eligible, "accuracy_assessed": assessed,
            "document_exact": bool(eligible and normalized == expected) if assessed else None,
            "order_insensitive_exact": bool(eligible and complete == len(expected) == len(normalized)) if assessed else None,
            "reference_rows": len(expected) if assessed else None, "predicted_rows": len(normalized),
            "detected_rows": len(matched) if assessed else None, "complete_rows": complete if assessed else None,
            "field_correct": field_correct if assessed else None, "matched_rows": details if assessed else None,
            "missing_reference_indices": missing if assessed else None, "extra_prediction_indices": extra if assessed else None,
            "invalid_numeric_fields": invalid, "arithmetic_error_indices": arithmetic,
            "parse_error": parse_error, "inference_error": error,
            "truncated": bool(prediction.get("truncated", False)),
            "parsed_rows": parsed, "parsed_csv": raw if parse_error is None else None}


def ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else 0.0}


def summarize(details, expected_documents):
    if len(details) != expected_documents:
        raise ValueError("Missing evaluations would change the fixed document denominator")
    references = sum(row["reference_rows"] for row in details)
    predictions = sum(row["predicted_rows"] for row in details)
    detected = sum(row["detected_rows"] for row in details)
    complete = sum(row["complete_rows"] for row in details)
    precision = detected / predictions if predictions else 0.0
    recall = detected / references if references else 0.0
    return {
        "documents": expected_documents,
        **{name: ratio(sum(bool(row[name]) for row in details), expected_documents)
           for name in ("csv_valid", "eligible_completion", "document_exact", "order_insensitive_exact")},
        "fields": {field: ratio(sum(row["field_correct"][field] for row in details), references)
                   for field in CSV_FIELDS},
        "row_precision": ratio(detected, predictions), "row_recall": ratio(detected, references),
        "row_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "complete_row_exact": ratio(complete, references),
        "complete_row_precision": ratio(complete, predictions),
        "missing_rows": sum(len(row["missing_reference_indices"]) for row in details),
        "extra_rows": sum(len(row["extra_prediction_indices"]) for row in details),
        "parse_failures": sum(row["parse_error"] is not None for row in details),
        "invalid_numeric_fields": sum(len(row["invalid_numeric_fields"]) for row in details),
        "arithmetic_errors": sum(len(row["arithmetic_error_indices"]) for row in details),
        "incorrect_rate": references - sum(row["field_correct"]["Rate"] for row in details),
        "incorrect_amount": references - sum(row["field_correct"]["Amount"] for row in details),
        "truncations": sum(row["truncated"] for row in details),
        "inference_errors": sum(bool(row["inference_error"]) for row in details),
    }
