/* Phase 9: real FastAPI invoice submission, source-aware results, client-side exports.
   All untrusted response text is inserted through textContent (not innerHTML). */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const fileInput = $('invoice-file');
  const dropzone = $('dropzone');
  const selected = $('selected-file');
  const form = $('claim-form');
  const submit = $('submit-btn');
  const message = $('message');
  const referenceSelect = $('reference-id');
  const MAX_FILE_BYTES = 8 * 1024 * 1024;
  let mode = 'csv';
  let chosenFile = null;
  let previewUrl = null;
  let lastResult = null;

  function showMessage(text, isError = false) {
    message.textContent = String(text);
    message.classList.remove('hidden');
    message.classList.toggle('error', isError);
  }
  function clearMessage() { message.classList.add('hidden'); message.textContent = ''; }
  function setBusy(busy) {
    submit.setAttribute('aria-busy', String(busy));
    submit.disabled = busy || (mode === 'demo' ? !referenceSelect.value : !chosenFile);
    submit.textContent = busy ? 'Processing invoice…' : 'Run adjudication →';
  }
  function cleanFile() {
    fileInput.value = '';
    chosenFile = null;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    $('preview').removeAttribute('src');
    $('preview').classList.add('hidden');
    selected.classList.add('hidden');
    setBusy(false);
  }
  function chooseFile(file) {
    if (!file) return;
    const ext = (file.name.split('.').pop() || '').toLowerCase();
    const valid = mode === 'image' ? ['jpg','jpeg','png'] : ['csv','json'];
    if (!valid.includes(ext)) { cleanFile(); showMessage(`Please select a ${valid.join(' / ')} file.`, true); return; }
    if (file.size > MAX_FILE_BYTES || file.size === 0) { cleanFile(); showMessage('File must be nonempty and no larger than 8 MB.', true); return; }
    cleanFile();
    clearMessage();
    chosenFile = file;
    $('file-name').textContent = file.name;
    $('file-meta').textContent = `${(file.size / 1024).toFixed(1)} KB · ${ext.toUpperCase()}`;
    if (mode === 'image') {
      previewUrl = URL.createObjectURL(file);
      $('preview').src = previewUrl;
      $('preview').classList.remove('hidden');
    }
    selected.classList.remove('hidden');
    setBusy(false);
  }
  function updateMode(next) {
    mode = next;
    cleanFile(); clearMessage();
    $('file-mode').classList.toggle('hidden', mode === 'demo');
    $('demo-mode').classList.toggle('hidden', mode !== 'demo');
    $('date-box').classList.toggle('hidden', mode === 'demo');
    const image = mode === 'image';
    fileInput.accept = image ? 'image/jpeg,image/png,.jpg,.jpeg,.png' : '.csv,.json,text/csv,application/json';
    dropzone.querySelector('strong').textContent = image ? 'Drop a JPG/PNG invoice here, or browse files' : 'Drop a CSV/JSON here, or browse files';
    $('drop-hint').textContent = image ? 'Requires an authenticated remote VLM server · Maximum 8 MB' : 'CSV: Item,Quantity,Rate,Amount · JSON: labeled invoice schema · Maximum 8 MB';
    $('mode-help').textContent = image ? 'Actual image extraction is unavailable until your GPU API is configured.' : mode === 'demo' ? 'This is a ground-truth component demo, not image extraction.' : 'CSV/JSON runs on the local machine without an LLM.';
    document.querySelectorAll('.mode-btn').forEach(b => {
      const active = b.dataset.mode === mode;
      b.classList.toggle('active', active);
      b.setAttribute('aria-pressed', String(active));
    });
    setBusy(false);
  }
  function money(amount) {
    if (amount === null || amount === undefined || amount === '') return 'Pending review';
    const value = Number(amount);
    return Number.isFinite(value) ? new Intl.NumberFormat('en-IN', {style:'currency',currency:'INR',minimumFractionDigits:2}).format(value) : String(amount);
  }
  function el(tag, content, cls = '') {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (content !== undefined && content !== null) node.textContent = String(content);
    return node;
  }
  function render(result) {
    const claim = result.claim || {};
    lastResult = result;
    $('empty-state').classList.add('hidden');
    $('result-content').classList.remove('hidden');
    $('status-line').textContent = `${claim.invoice_id || 'Invoice'} · ${claim.claim_status === 'finalized' ? 'Finalized' : 'Manual review required'} · ${claim.currency || 'INR'}`;
    $('metric-billed').textContent = money(claim.billed_total);
    $('metric-eligible').textContent = money(claim.eligible_total);
    $('metric-copay').textContent = money(claim.copay_amount);
    $('metric-final').textContent = claim.final_reimbursement === null ? 'Not finalized' : money(claim.final_reimbursement);
    $('meta-input').textContent = `Input: ${result.extraction_source || 'unknown'}`;
    $('meta-rag').textContent = `Retrieval: ${result.rag_status || 'not_run'}`;
    $('meta-policy').textContent = `Policy: ${claim.policy_id || 'N/A'}`;
    const warningText = (result.warnings || []).join('\n');
    $('result-warning').textContent = warningText;
    $('result-warning').classList.toggle('hidden', !warningText);
    const rows = $('decision-rows');
    rows.replaceChildren();
    for (const decision of (claim.decisions || [])) {
      const tr = document.createElement('tr');
      tr.append(el('td', decision.item), el('td', decision.quantity),
        el('td', money(decision.billed_amount)), el('td', money(decision.approved_amount)));
      const tdStatus = document.createElement('td');
      tdStatus.append(el('span', decision.status, `decision-state ${decision.status}`));
      tr.append(tdStatus, el('td', decision.rule_id || 'Review'), el('td', decision.explanation));
      rows.append(tr);
    }
    $('csv-preview').textContent = result.extracted_csv || 'CSV is not available for this run.';
    $('result-details').replaceChildren(el('pre', JSON.stringify({
      input_mode: result.input_mode,
      model: result.metadata?.vlm_model || null,
      model_latency_ms: result.metadata?.vlm_latency_ms || null,
      trace: result.trace || [],
      notes: claim.notes || [],
      warnings: result.warnings || [],
    }, null, 2)));
    $('results').scrollIntoView({behavior:'smooth',block:'start'});
  }
  function download(name, type, content) {
    const blob = new Blob([content], {type});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function explainFailure(data, statusCode) {
    const detail = data?.detail;
    if (typeof detail === 'string') return detail;
    if (detail && typeof detail === 'object') {
      const errors = detail.workflow?.errors;
      if (Array.isArray(errors) && errors.length) return errors.join('\n');
      if (detail.message) return detail.message;
    }
    return `Request failed (HTTP ${statusCode}).`;
  }
  async function loadExamples() {
    try {
      const response = await fetch('/api/examples');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      for (const item of data.examples || []) {
        const opt = el('option', item.label);
        opt.value = item.id;
        referenceSelect.append(opt);
      }
    } catch (_) { showMessage('Reference examples could not be loaded; upload a CSV instead.', true); }
  }
  document.querySelectorAll('.mode-btn').forEach(btn => btn.addEventListener('click', () => updateMode(btn.dataset.mode)));
  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } });
  fileInput.addEventListener('change', () => chooseFile(fileInput.files[0]));
  for (const ev of ['dragover', 'dragenter']) dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.add('dragover'); });
  for (const ev of ['dragleave','drop']) dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.remove('dragover'); });
  dropzone.addEventListener('drop', e => chooseFile(e.dataTransfer.files[0]));
  $('remove-file').addEventListener('click', cleanFile);
  referenceSelect.addEventListener('change', () => setBusy(false));
  form.addEventListener('submit', async e => {
    e.preventDefault();
    clearMessage();
    setBusy(true);
    try {
      let response;
      if (mode === 'demo') {
        response = await fetch('/api/adjudicate/demo', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body:JSON.stringify({example_id:referenceSelect.value,use_milvus:$('use-milvus').checked})
        });
      } else {
        if (!chosenFile) throw new Error('Choose a file first.');
        const data = new FormData();
        data.append('file', chosenFile);
        data.append('invoice_date', $('invoice-date').value);
        data.append('use_milvus', $('use-milvus').checked ? 'true' : 'false');
        response = await fetch('/api/adjudicate', {method:'POST', body:data});
      }
      const result = await response.json();
      if (!response.ok) throw new Error(explainFailure(result, response.status));
      render(result);
      showMessage('LangGraph completed. Review the source, rules and final amount below.');
    } catch (err) {
      showMessage(err.message || 'Unable to process the invoice.', true);
    } finally { setBusy(false); }
  });
  $('download-json').addEventListener('click', () => {
    if (lastResult) download('claim_decision.json','application/json', JSON.stringify(lastResult,null,2)+'\n');
  });
  $('download-csv').addEventListener('click', () => {
    if (lastResult?.extracted_csv) download('invoice_extraction.csv','text/csv',lastResult.extracted_csv);
  });
  $('copy-csv').addEventListener('click', async () => {
    if (lastResult?.extracted_csv) {
      try { await navigator.clipboard.writeText(lastResult.extracted_csv); showMessage('CSV copied.'); }
      catch (_) { showMessage('Clipboard unavailable; use Download CSV.', true); }
    }
  });
  window.addEventListener('pagehide', () => { if (previewUrl) URL.revokeObjectURL(previewUrl); });
  updateMode('csv');
  loadExamples();
})();
