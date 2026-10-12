"""Facades for the preserved pipeline and request-scoped two-agent claim graph."""
from pathlib import Path

from src.components.agentic_workflow.graph import create_graph


def run_claim_workflow(*, invoice_json=None, csv_file=None, csv_text=None,
                       image=None, invoice_date=None, invoice_id=None,
                       with_milvus=False, strict_rag=False, remote_vlm=False,
                       retriever=None, remote_client=None, policy=None, resolver=None):
    """Return the final JSON-compatible workflow result, not internal state."""
    sources = {"invoice_json": invoice_json, "csv_file": csv_file,
               "csv_text": csv_text, "image": image}
    provided = [name for name, value in sources.items() if value is not None]
    if len(provided) != 1:
        raise ValueError("Supply exactly one of invoice_json, csv_file, csv_text, image")
    mode = provided[0]
    if mode != "invoice_json" and not invoice_date:
        raise ValueError("invoice_date YYYY-MM-DD is required for CSV and image input")
    state = {
        "input_mode": "csv_file" if mode == "csv_file" else mode,
        "invoice_json_path": str(invoice_json) if invoice_json is not None else "",
        "csv_path": str(csv_file) if csv_file is not None else "",
        "csv_text": str(csv_text) if csv_text is not None else "",
        "image_path": str(image) if image is not None else "",
        "invoice_date": str(invoice_date) if invoice_date else "",
        "invoice_id": invoice_id or (Path(sources[mode]).stem if mode != "csv_text" else "provided_invoice"),
        "use_milvus": with_milvus,
        "strict_rag": strict_rag,
        "use_remote_vlm": remote_vlm,
        "warnings": [], "errors": [], "trace": [], "metadata": {},
    }
    graph = create_graph(remote_client=remote_client, retriever=retriever,
                         policy=policy, resolver=resolver)
    state = graph.invoke(state)
    return state["result"]


def initialize_claim_state(*, image=None, csv_file=None, csv_text=None, invoice_date=None,
                           request_id=None, thread_id=None, max_retries=1, request_timeout=300):
    """Fresh typed state for one request; no inherited claim data or global lists."""
    from datetime import date
    import math
    import re
    import time
    from uuid import uuid4
    from src.components.inference.vlm_client import ADAPTER_SERVED_MODEL, BASE_SERVED_MODEL

    provided = [key for key, value in {"image": image, "csv_file": csv_file, "csv_text": csv_text}.items() if value is not None]
    if len(provided) != 1:
        raise ValueError("Supply exactly one image, CSV file or CSV text")
    if not invoice_date:
        raise ValueError("Invoice date is required; never infer it from the current date")
    when = date.fromisoformat(str(invoice_date))
    if max_retries not in (0, 1) or not math.isfinite(request_timeout) or request_timeout <= 0:
        raise ValueError("Retry budget must be 0 or 1 and request timeout must be positive")
    request_id = request_id or uuid4().hex
    for identifier in (request_id, thread_id):
        if identifier is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", identifier):
            raise ValueError("Invalid request/thread identifier")
    mode = provided[0]
    started = time.monotonic()
    return {"request_id": request_id, "thread_id": thread_id, "input_mode": mode,
            "image_path": str(image) if image is not None else "", "csv_path": str(csv_file) if csv_file is not None else "",
            "csv_text": csv_text if csv_text is not None else "", "invoice_date": when.isoformat(),
            "invoice_id": Path(image or csv_file).stem if image or csv_file else request_id,
            "workflow_status": "RUNNING", "csv_validated": False, "retry_counters": {}, "retry_node": None,
            "max_retries": max_retries, "execution_trace": [], "trace": [], "errors": [], "warnings": [], "metadata": {},
            "model_ids": {"extraction": ADAPTER_SERVED_MODEL, "agent_1": BASE_SERVED_MODEL, "agent_2": BASE_SERVED_MODEL},
            "started_monotonic": started, "deadline_monotonic": started+request_timeout}


def run_claim_graph(*, image=None, csv_file=None, csv_text=None, invoice_date=None,
                    endpoint="http://127.0.0.1:8000/v1", request_id=None, thread_id=None,
                    max_retries=1, request_timeout=300, checkpoint=False, verbose=False, **services):
    """Invoke the real compiled two-agent graph; integrations are injected only for tests."""
    from src.components.agentic_workflow.graph import create_claim_graph
    state = initialize_claim_state(image=image, csv_file=csv_file, csv_text=csv_text,
        invoice_date=invoice_date, request_id=request_id, thread_id=thread_id,
        max_retries=max_retries, request_timeout=request_timeout)
    graph = create_claim_graph(endpoint=endpoint, checkpoint=checkpoint, verbose=verbose, **services)
    config = {"recursion_limit": 24}
    if checkpoint:
        # A new namespace per request prevents accidental reuse of mutable claim state.
        config["configurable"] = {"thread_id": f"{thread_id or 'claim'}:{state['request_id']}"}
    return graph.invoke(state, config=config)["result"]
