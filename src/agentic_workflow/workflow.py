"""Small facade for invoking Phase 8 LangGraph from CLI or future FastAPI.

No background jobs. A caller either supplies structured text or explicitly
opts into remote VLM image extraction (the GPU server is currently on hold).
"""
from pathlib import Path

from src.agentic_workflow.graph import create_graph


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
