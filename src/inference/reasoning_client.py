"""Strict text JSON calls to the existing local qwen-base vLLM server."""
from dataclasses import replace
import json
import time

from pydantic import ValidationError
from src.agents.contracts import AgentFormatError, CandidateSelections, FactSelections
from src.inference.remote_client import RemoteVLMClient, RemoteVLMError
from src.inference.vlm_client import BASE_MODEL_ID, BASE_SERVED_MODEL, VLMConfig


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Nonfinite JSON number")


class TextReasoningClient:
    """Reuse Phase 6 HTTP transport; no Torch imports, API key or model downloads."""
    def __init__(self, endpoint="http://127.0.0.1:8000/v1", *, timeout=120,
                 max_tokens=768, context_length=2048, http_client=None):
        endpoint = endpoint.rstrip("/")
        if not endpoint.endswith("/v1"):
            endpoint += "/v1"
        self.config = VLMConfig(mode="base", model=BASE_SERVED_MODEL, endpoint=endpoint,
                                timeout=timeout, max_tokens=max_tokens).transport_config()
        if context_length < 128 or max_tokens >= context_length:
            raise ValueError("Invalid reasoning context/output token budget")
        self.context_length = context_length
        self.http_client = http_client

    def _send(self, method, path, payload, deadline, *, api_root=False):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RemoteVLMError("Claim request deadline exceeded")
        config = replace(self.config, timeout_seconds=min(self.config.timeout_seconds, remaining))
        return RemoteVLMClient(config, self.http_client)._request(method, path, payload, api_root=api_root)

    def generate(self, instruction, data, schema, *, deadline):
        started = time.monotonic()
        cards = self._send("GET", "/models", None, deadline).get("data", [])
        if not isinstance(cards, list):
            raise RemoteVLMError("Malformed served model list")
        card = next((card for card in cards if isinstance(card, dict) and card.get("id") == BASE_SERVED_MODEL), None)
        if not card or card.get("parent") or card.get("root") != BASE_MODEL_ID:
            raise RemoteVLMError("qwen-base is absent or is not the expected base model")
        messages = [{"role": "system", "content": instruction},
                    {"role": "user", "content": json.dumps(data, ensure_ascii=False, separators=(",", ":"))}]
        tokens = self._send("POST", "/tokenize", {"model": BASE_SERVED_MODEL,
            "messages": messages, "add_generation_prompt": True}, deadline, api_root=True)
        count, server_limit = tokens.get("count"), tokens.get("max_model_len")
        if type(count) is not int or type(server_limit) is not int or count < 1:
            raise RemoteVLMError("vLLM did not return a valid chat token count")
        if count + self.config.max_tokens > min(self.context_length, server_limit):
            raise RemoteVLMError("Reasoning context exceeds the server budget; reduce rows/evidence or output tokens")
        response = self._send("POST", "/chat/completions", {
            "model": BASE_SERVED_MODEL, "messages": messages, "temperature": 0,
            "seed": 42, "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "strict": True, "schema": schema.model_json_schema()}},
        }, deadline)
        if response.get("model") != BASE_SERVED_MODEL:
            raise RemoteVLMError("Reasoning response selected an incorrect served model")
        try:
            choice = response["choices"][0]
            raw = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AgentFormatError("Missing assistant JSON") from exc
        if choice.get("finish_reason") != "stop":
            raise RemoteVLMError("Reasoning response is truncated or did not finish normally")
        try:
            value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
            output = schema.model_validate(value)
        except (ValueError, TypeError, ValidationError) as exc:
            error = AgentFormatError("Invalid reasoning JSON/schema")
            if schema in (CandidateSelections, FactSelections):
                location = exc.errors(include_input=False, include_context=False)[0]["loc"] if isinstance(exc, ValidationError) else ()
                field = location[-1] if location and location[-1] in {
                    "item_index", "candidate_index", "suggested_status", "basis", "items",
                    "status", "fact_indices", "concerns"} else "response_schema"
                index = location[1] if len(location) > 1 and type(location[1]) is int else None
                error.diagnostics = {"category": "invalid_json_or_schema", "item_index": index,
                    "field": field, "candidate_index": None, "sanitized_response": None}
            raise error from exc
        usage = response.get("usage", {})
        usage = {key: value for key, value in usage.items() if type(value) is int and value >= 0} if isinstance(usage, dict) else {}
        return output, {"model": BASE_SERVED_MODEL, "latency_ms": round((time.monotonic()-started)*1000),
                        "prompt_tokens": count, "usage": usage}
