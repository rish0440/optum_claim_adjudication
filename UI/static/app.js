"use strict";
const $ = id => document.getElementById(id);
let selected = null, previewUrl = null, activeJob = null, busy = false;
const money = value => value == null ? "Not finalized" : `INR ${Number(value).toLocaleString("en-IN", {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
function message(text, error = false) { $("message").textContent = text; $("message").className = `message${error ? " error" : ""}`; }
function selectFile(file) {
  if (busy) return;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  selected = null; $("selected-file").classList.add("hidden"); $("submit-btn").disabled = true;
  if (!file) return;
  if (!/\.(jpe?g|png)$/i.test(file.name) || !["image/jpeg", "image/png"].includes(file.type)) return message("Choose a JPG, JPEG or PNG image.", true);
  if (file.size > 8 * 1024 * 1024) return message("Maximum image size is 8 MB.", true);
  selected = file; previewUrl = URL.createObjectURL(file); $("preview").src = previewUrl;
  $("file-name").textContent = file.name; $("file-meta").textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
  $("selected-file").classList.remove("hidden"); $("message").classList.add("hidden"); $("submit-btn").disabled = false;
}
$("browse").onclick = e => { e.stopPropagation(); if (!busy) $("invoice-file").click(); };
$("dropzone").onclick = e => { if (e.target !== $("invoice-file") && !busy) $("invoice-file").click(); };
$("dropzone").onkeydown = e => { if (e.target === $("dropzone") && ["Enter", " "].includes(e.key)) { e.preventDefault(); if (!busy) $("invoice-file").click(); } };
$("invoice-file").onchange = e => selectFile(e.target.files[0]);
$("remove-file").onclick = () => { selectFile(null); $("invoice-file").value = ""; };
for (const event of ["dragover", "dragleave", "drop"]) $("dropzone").addEventListener(event, e => {
  e.preventDefault(); $("dropzone").classList.toggle("dragover", event === "dragover");
  if (event === "drop") { if (e.dataTransfer.files.length !== 1) return message("Upload one invoice image at a time.", true); selectFile(e.dataTransfer.files[0]); }
});
async function api(url, options) {
  const response = await fetch(url, options), data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Request failed. Check the local API service.");
  return data;
}
function el(tag, text, className) { const node = document.createElement(tag); node.textContent = text ?? ""; if (className) node.className = className; return node; }
function render(job) {
  const report = job.result, claim = report?.claim;
  $("results").classList.remove("hidden");
  $("status-line").textContent = report?.status === "REVIEW_REQUIRED" ? "Review Required — Final reimbursement not yet determined." : report?.status === "COMPLETED" ? "Assessment completed" : job.message;
  $("result-meta").replaceChildren(...[`Request: ${job.request_id}`, `Invoice date: ${job.invoice_date?.invoice_date || "Unavailable"}`, "Date: model-extracted, unverified", `Policy: ${claim?.policy_id || "Unavailable"}`, `Status: ${report?.status || job.state}`, `Agent reviewed: ${report?.reviewed ? "Yes" : "No"}`].map(x => el("span", x)));
  for (const id of ["metrics", "decision-rows", "reasoning"]) $(id).replaceChildren();
  const warnings = [...(report?.warnings || []), ...(report?.errors || []), ...(claim?.notes || []), ...(job.report_error ? [job.report_error] : [])];
  if (!report) warnings.push(job.message);
  $("result-warning").textContent = warnings.join("\n"); $("result-warning").classList.toggle("hidden", !warnings.length);
  if (claim) {
    const finalized = report.status === "COMPLETED" && claim.claim_status === "finalized";
    for (const [label, value] of [["Billed", claim.billed_total], ["Eligible (provisional if pending)", claim.eligible_total], ["Disallowed", claim.disallowed_total], ["Pending review", claim.pending_review_total], [`Copay (${claim.copay_rate_percent}%)`, claim.copay_amount], ["Final reimbursement", finalized ? claim.final_reimbursement : null]]) {
      const card = el("div", "", "metric"); card.append(el("span", label), el("strong", money(value))); $("metrics").append(card);
    }
    claim.decisions.forEach((item, index) => {
      const review = report.reviewed ? report.agent_2?.items.find(x => x.item_index === index) : null;
      const row = document.createElement("tr");
      for (const value of [item.item, item.quantity, money(item.billed_amount), money(item.approved_amount), money(item.disallowed_amount), money(item.pending_review_amount), item.status, item.rule_id || "Pending evidence", review?.explanation || item.explanation]) row.append(el("td", value));
      $("decision-rows").append(row);
      const section = el("section", "", "item-reasoning"); section.append(el("h3", item.item));
      const assessment = report.agent_1?.items.find(x => x.item_index === index);
      if (assessment) section.append(el("p", `Agent 1: ${assessment.reason}`));
      section.append(el("p", review ? `Verified Agent 2: ${review.explanation}` : `Deterministic explanation (Agent 2 unreviewed): ${item.explanation}`));
      if (item.review_note) section.append(el("p", `Evidence required: ${item.review_note}`, "result-warning"));
      const ids = new Set([item.rule_id, ...(assessment?.policy_ids || []), ...(review?.policy_ids || [])]);
      for (const fact of report.evidence[index] || []) if (ids.has(fact.rule_id)) section.append(el("p", `${fact.rule_id}: ${fact.conditions || fact.text}`));
      $("reasoning").append(section);
    });
  }
  $("csv-preview").textContent = report?.extracted_csv || "Unavailable";
  $("trace").textContent = JSON.stringify(report?.trace || [], null, 2); $("models").textContent = JSON.stringify(report?.metadata || {}, null, 2);
  $("download-pdf").classList.toggle("hidden", !job.report_available); $("download-pdf").href = `/api/claims/${job.request_id}/report.pdf`;
  $("download-json").disabled = !report; $("download-csv").disabled = !report?.extracted_csv;
}
$("claim-form").onsubmit = async e => {
  e.preventDefault(); if (!selected || busy) return;
  busy = true; $("submit-btn").disabled = true; $("submit-btn").setAttribute("aria-busy", "true"); $("processing").classList.remove("hidden"); $("results").classList.add("hidden");
  try {
    const form = new FormData(); form.append("file", selected);
    const queued = await api("/api/claims/analyze", {method: "POST", body: form}); message("Claim queued. Waiting for the actual workflow result…");
    while (true) {
      await new Promise(resolve => setTimeout(resolve, 2000)); activeJob = await api(queued.status_url); message(activeJob.message);
      if (!["queued", "processing"].includes(activeJob.state)) { render(activeJob); message(activeJob.message, activeJob.state === "failed"); break; }
    }
  } catch (error) { message(error.message, true); }
  finally { busy = false; $("submit-btn").disabled = !selected; $("submit-btn").removeAttribute("aria-busy"); $("processing").classList.add("hidden"); }
};
function download(text, suffix, mime) { const url = URL.createObjectURL(new Blob([text], {type: mime})); const link = document.createElement("a"); link.href = url; link.download = `claim-${activeJob.request_id}.${suffix}`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
$("download-json").onclick = () => download(JSON.stringify(activeJob, null, 2), "json", "application/json");
$("download-csv").onclick = () => download(activeJob.result.extracted_csv, "csv", "text/csv");
