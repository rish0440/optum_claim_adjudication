"""A4 assessment reports built only from the graph's authoritative output."""
from datetime import datetime, timezone
from decimal import Decimal
from xml.sax.saxutils import escape


def money(value):
    return "NOT FINALIZED" if value is None else f"INR {Decimal(str(value)):,.2f}"


def generate_claim_pdf(report, date_metadata, destination):
    # Lazy import: a missing PDF dependency must not disable existing APIs.
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_RIGHT
    from reportlab.lib.units import mm
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    navy, teal = colors.HexColor("#172943"), colors.HexColor("#147b84")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("Cell", fontName="Helvetica", fontSize=8, leading=11, spaceAfter=0))
    styles.add(ParagraphStyle("Amount", parent=styles["Cell"], alignment=TA_RIGHT))
    styles["Title"].textColor = navy
    styles["Heading2"].textColor = teal
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 13
    def p(text, style="BodyText"):
        return Paragraph(escape(str(text)), styles[style])
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    claim = report.get("claim") or {}
    story = [p("Claim Adjudication Copilot", "Title"), p("Claim Assessment Report", "Heading2")]
    for label, value in (("Request ID", report["request_id"]), ("Generated", stamp),
                         ("Invoice date", date_metadata.get("invoice_date", "Unavailable")),
                         ("Date source", "Model-extracted; not independently verified"),
                         ("Policy ID", claim.get("policy_id", "Unavailable")),
                         ("Status", report["status"]), ("Agent reviewed", "Yes" if report.get("reviewed") else "No")):
        story.append(p(f"{label}: {value}"))
    story.append(p("Financial summary", "Heading2"))
    finalized = report["status"] == "COMPLETED" and claim.get("claim_status") == "finalized"
    if report["status"] == "FAILED":
        story.append(p("Workflow failed — No finalized reimbursement."))
    elif not finalized:
        story.append(p("Review Required — Final reimbursement not yet determined."))
    summary = [[p(label), p(money(claim.get(key) if key != "final_reimbursement" or finalized else None), "Amount")] for label, key in (
        ("Total billed", "billed_total"), ("Eligible (provisional if pending)", "eligible_total"),
        ("Disallowed", "disallowed_total"), ("Pending review", "pending_review_total"),
        (f"Copay ({claim.get('copay_rate_percent', 'unavailable')}%)", "copay_amount"),
        ("Final reimbursement", "final_reimbursement"))]
    story.append(Table(summary, colWidths=[115*mm, 59*mm], style=TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#f3f7fa")),
        ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7)])))
    story.append(p("Item-level decisions", "Heading2"))
    rows = [[p(x, "Cell") for x in ("Item", "Billed", "Approved", "Disallowed", "Pending", "Decision / rule")]]
    for item in claim.get("decisions", []):
        rows.append([p(item["item"], "Cell"), *[p(money(item.get(key)), "Amount") for key in
            ("billed_amount", "approved_amount", "disallowed_amount", "pending_review_amount")],
            p(f"{item['status']} / {item.get('rule_id') or 'Pending evidence'}", "Cell")])
    table = Table(rows, colWidths=[43*mm, 25*mm, 25*mm, 25*mm, 25*mm, 31*mm], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e6f2f2")),
        ("VALIGN", (0,0), (-1,-1), "TOP"), ("LINEBELOW", (0,0), (-1,0), .5, teal),
        ("LINEBELOW", (0,1), (-1,-1), .25, colors.HexColor("#dce4ec")),
        ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7)]))
    story.extend([table, Spacer(1, 4*mm), p("Policy analysis", "Heading2")])
    reviews = {row["item_index"]: row for row in (report.get("agent_2") or {}).get("items", [])}
    suggestions = {row["item_index"]: row for row in (report.get("agent_1") or {}).get("items", [])}
    for index, item in enumerate(claim.get("decisions", [])):
        story.append(p(f"{index+1}. {item['item']} — {item['status']}", "Heading3"))
        story.append(p(item.get("explanation", "")))
        if item.get("review_note"):
            story.append(p("Evidence required: " + item["review_note"]))
        assessment = suggestions.get(index)
        if assessment:
            story.append(p("Agent 1 assessment: " + assessment["reason"]))
        if report.get("reviewed") and index in reviews:
            story.append(p("Verified Agent 2 review: " + reviews[index]["explanation"]))
        selected = set((assessment or {}).get("policy_ids", [])) | set(reviews.get(index, {}).get("policy_ids", []))
        if item.get("rule_id"):
            selected.add(item["rule_id"])
        for evidence in report.get("evidence", {}).get(str(index), report.get("evidence", {}).get(index, [])):
            if evidence.get("rule_id") in selected:
                story.append(p(f"{evidence['rule_id']}: {evidence.get('conditions') or evidence.get('text', '')}"))
    story.append(p("Review findings", "Heading2"))
    if not report.get("reviewed"):
        story.append(p("Agent 2 review is unverified or unavailable. Deterministic results remain authoritative."))
    for text in [*claim.get("notes", []), *report.get("warnings", []), *report.get("errors", [])]:
        story.append(p(text))
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(navy)
        canvas.drawString(18*mm, 13*mm, "Prototype assessment; subject to evidence verification and insurer review.")
        canvas.drawString(18*mm, 9*mm, stamp)
        canvas.drawRightString(192*mm, 9*mm, f"Page {doc.page}")
        canvas.restoreState()
    SimpleDocTemplate(str(destination), pagesize=A4, leftMargin=18*mm, rightMargin=18*mm,
        topMargin=18*mm, bottomMargin=22*mm, title="Claim Assessment Report").build(story, onFirstPage=footer, onLaterPages=footer)
