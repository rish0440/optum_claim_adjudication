"""Strict JSON calls with explicit Phase 8 / experimental text reasoning routing."""
from dataclasses import replace
import json
import logging
from pathlib import Path
import time

from pydantic import ValidationError
from src.components.agentic_workflow.contracts import AgentFormatError, CandidateSelections, FactSelections
from src.config.reasoning import reasoning_route
from src.components.inference.remote_client import RemoteVLMClient, RemoteVLMConfig, RemoteVLMError

logger = logging.getLogger(__name__)


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
        self.route = reasoning_route(endpoint)
        self.config = RemoteVLMConfig(base_url=self.route.endpoint, api_key="", model=self.route.model,
                                      timeout_seconds=timeout, max_tokens=max_tokens, require_api_key=False)
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

    def preflight(self, *, deadline):
        """Check the selected server identity; never switch endpoints on failure."""
        cards = self._send("GET", "/models", None, deadline).get("data", [])
        if not isinstance(cards, list):
            raise RemoteVLMError("Malformed served model list")
        matches = [card for card in cards if isinstance(card, dict) and card.get("id") == self.config.model]
        if len(matches) != 1:
            raise RemoteVLMError("Selected reasoning model is absent or ambiguous")
        card = matches[0]
        if self.route.backend == "text_dpo":
            bases = [row for row in cards if isinstance(row, dict) and row.get("id") == "reasoning-base"]
            if (len(bases) != 1 or bases[0].get("parent") or bases[0].get("root") != self.route.base_model_id
                    or card.get("parent") != "reasoning-base" or not isinstance(card.get("root"), str)
                    or not Path(card["root"]).is_absolute()
                    or Path(card["root"]).resolve() != self.route.adapter_path):
                raise RemoteVLMError("DPO server adapter/base identity does not match the successful checkpoint")
        elif card.get("parent") or card.get("root") != self.route.base_model_id:
            raise RemoteVLMError("Selected reasoning model is not the expected unadapted backbone")
        return self.route.diagnostics()

    def generate(self, instruction, data, schema, *, deadline):
        started = time.monotonic()
        self.preflight(deadline=deadline)
        logger.info("Reasoning request backend=%s endpoint=%s model=%s schema=%s",
                    self.route.backend, self.config.base_url, self.config.model, schema.__name__)
        messages = [{"role": "system", "content": instruction},
                    {"role": "user", "content": json.dumps(data, ensure_ascii=False, separators=(",", ":"))}]
        tokens = self._send("POST", "/tokenize", {"model": self.config.model,
            "messages": messages, "add_generation_prompt": True}, deadline, api_root=True)
        count, server_limit = tokens.get("count"), tokens.get("max_model_len")
        if type(count) is not int or type(server_limit) is not int or count < 1:
            raise RemoteVLMError("vLLM did not return a valid chat token count")
        if count + self.config.max_tokens > min(self.context_length, server_limit):
            raise RemoteVLMError("Reasoning context exceeds the server budget; reduce rows/evidence or output tokens")
        response = self._send("POST", "/chat/completions", {
            "model": self.config.model, "messages": messages, "temperature": 0,
            "seed": 42, "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "strict": True, "schema": schema.model_json_schema()}},
        }, deadline)
        if response.get("model") != self.config.model:
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
        logger.info("Reasoning response served_model=%s schema=%s endpoint=%s",
                    response["model"], schema.__name__, self.config.base_url)
        return output, {**self.route.diagnostics(), "latency_ms": round((time.monotonic()-started)*1000),
                        "prompt_tokens": count, "usage": usage}
