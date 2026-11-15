"""Adapter for OpenAI-compatible chat completion servers (v0.5.0).

Works with any server that exposes ``POST {base_url}/chat/completions`` and, optionally,
``GET {base_url}/models``: vLLM, LM Studio, llama.cpp server, Ollama's OpenAI-compatible
endpoint, and hosted APIs that follow the same format.

Security:
- The API key is read from an environment variable at request time. It is never stored on the
  adapter, in ``inference_config``, in caches, logs, run directories or reports.
- Endpoints are validated with :func:`openbench.security.validate_endpoint_url`; plain HTTP is
  only allowed for loopback hosts unless ``allow_remote_http=True``.
"""

from __future__ import annotations

import email.utils
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import timezone
from typing import Any

from openbench.adapters.base import BaseModelAdapter, GenerationConfig, ProviderHealthStatus
from openbench.dataset import DatasetRecord
from openbench.evaluator import ResponseItem
from openbench.exceptions import ConfigurationError, OpenBenchError, ProviderRateLimitError
from openbench.security import sanitize_error_message, validate_endpoint_url

DEFAULT_OPENAI_COMPATIBLE_ENDPOINT = "http://localhost:8000/v1"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"

#: Transport signature for tests: ``(method, url, payload, headers, timeout) -> (status, headers, body_text)``.
TransportFn = Callable[[str, str, "dict[str, Any] | None", "dict[str, str]", float], "tuple[int, dict[str, str], str]"]


class OpenAICompatibleConnectionError(OpenBenchError):
    """Raised when an OpenAI-compatible server cannot be reached or returns an error."""


def parse_retry_after(value: str | None, now: float | None = None) -> float | None:
    """Parse an HTTP ``Retry-After`` header (seconds or HTTP date) into seconds, or ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError):
            return None
        if parsed is None:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        reference = now if now is not None else time.time()
        seconds = parsed.timestamp() - reference
    if seconds != seconds or seconds < 0:  # NaN or in the past
        return 0.0
    return seconds


class OpenAICompatibleAdapter(BaseModelAdapter):
    """Adapter for servers implementing the OpenAI ``/v1/chat/completions`` API."""

    provider_name = "openai_compatible"
    description = (
        "OpenAI-compatible chat completions API (vLLM, LM Studio, llama.cpp server, hosted APIs)."
    )
    is_experimental = False
    is_mock = False
    supports_generation_seed = True

    def __init__(
        self,
        model_name: str,
        model_version: str = "latest",
        endpoint: str = DEFAULT_OPENAI_COMPATIBLE_ENDPOINT,
        api_key_env: str | None = DEFAULT_API_KEY_ENV,
        generation_config: GenerationConfig | None = None,
        timeout_seconds: float = 30.0,
        allow_remote_http: bool = False,
        system_prompt: str | None = None,
        transport_fn: TransportFn | None = None,
    ) -> None:
        validated = validate_endpoint_url(endpoint, allow_remote_http=allow_remote_http)
        inference: dict[str, Any] = {"endpoint": validated, "api": "chat.completions"}
        if system_prompt:
            inference["system_prompt"] = system_prompt
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            generation_config=generation_config,
            timeout_seconds=timeout_seconds,
            inference_config=inference,
        )
        if api_key_env is not None and (not isinstance(api_key_env, str) or not api_key_env.strip()):
            raise ConfigurationError("api_key_env must be a non-empty environment variable name or None")
        self.endpoint = validated
        # Only the variable *name* is kept; the value is read per request and never stored.
        self._api_key_env = api_key_env.strip() if api_key_env else None
        self.system_prompt = system_prompt
        self._transport_fn = transport_fn

    # -- HTTP ---------------------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key_env:
            key = os.environ.get(self._api_key_env)
            if key:
                headers["Authorization"] = f"Bearer {key}"
        return headers

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        url = f"{self.endpoint}{path}"
        headers = self._headers()
        if self._transport_fn is not None:
            status, resp_headers, body = self._transport_fn(method, url, payload, headers, self.timeout_seconds)
            return self._handle_response(url, status, resp_headers or {}, body)
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
                return self._handle_response(url, resp.status, dict(resp.headers.items()), body)
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = str(exc.reason)
            return self._handle_response(url, exc.code, dict(exc.headers.items()) if exc.headers else {}, body)
        except TimeoutError as exc:
            raise TimeoutError(f"Request to {url} timed out after {self.timeout_seconds}s") from exc
        except urllib.error.URLError as exc:
            reason = sanitize_error_message(str(exc.reason))
            if "timed out" in reason.lower():
                raise TimeoutError(f"Request to {url} timed out after {self.timeout_seconds}s") from exc
            raise OpenAICompatibleConnectionError(
                f"Could not connect to the OpenAI-compatible server at {self.endpoint} ({reason}). "
                "Check that the server is running and that --endpoint includes the /v1 prefix if the server needs it."
            ) from exc

    @staticmethod
    def _error_detail(body: str) -> str:
        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            return sanitize_error_message(str(body)[:300])
        if isinstance(parsed, dict):
            err = parsed.get("error")
            if isinstance(err, dict) and err.get("message"):
                return sanitize_error_message(str(err["message"])[:300])
            if isinstance(err, str):
                return sanitize_error_message(err[:300])
        return sanitize_error_message(str(body)[:300])

    def _handle_response(self, url: str, status: int, headers: dict[str, str], body: str) -> Any:
        lower = {str(k).lower(): v for k, v in headers.items()}
        if status == 429:
            retry_after = parse_retry_after(lower.get("retry-after"))
            raise ProviderRateLimitError(
                f"Rate limited by {url} (HTTP 429): {self._error_detail(body)}",
                retry_after_seconds=retry_after,
            )
        if status in (401, 403):
            hint = (
                f" Set the API key in the {self._api_key_env} environment variable."
                if self._api_key_env
                else ""
            )
            raise OpenAICompatibleConnectionError(
                f"Request to {url} was rejected (HTTP {status}): {self._error_detail(body)}.{hint}"
            )
        if status >= 400:
            raise OpenAICompatibleConnectionError(
                f"Request to {url} failed with HTTP {status}: {self._error_detail(body)}"
            )
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise OpenAICompatibleConnectionError(
                f"Response from {url} was not valid JSON: {sanitize_error_message(str(body)[:200])}"
            ) from exc

    # -- Provider interface ---------------------------------------------------------------------
    def check_connection(self) -> ProviderHealthStatus:
        """List models at ``GET {endpoint}/models`` and report whether the configured model is served."""
        details: dict[str, Any] = {"endpoint": self.endpoint, "model_name": self.model_name}
        try:
            payload = self._request("GET", "/models")
        except Exception as exc:
            return ProviderHealthStatus(
                provider_name=self.provider_name,
                available=False,
                message=sanitize_error_message(str(exc)),
                details=details,
            )
        data = payload.get("data", []) if isinstance(payload, dict) else []
        names = [str(m.get("id", "")) for m in data if isinstance(m, dict)]
        found = self.model_name in names
        details.update({"model_listed": found, "available_models": names})
        msg = (
            f"Connected to {self.endpoint}; model '{self.model_name}' is available."
            if found
            else f"Connected to {self.endpoint}, but model '{self.model_name}' is not in its model list ({names})."
        )
        return ProviderHealthStatus(provider_name=self.provider_name, available=True, message=msg, details=details)

    def build_payload(self, record: DatasetRecord) -> dict[str, Any]:
        """Return the JSON request body for ``record`` (no credentials)."""
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": record.prompt})
        cfg = self.generation_config
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "top_p": cfg.top_p,
            "stream": False,
        }
        if cfg.seed is not None:
            payload["seed"] = cfg.seed
        if cfg.stop_sequences:
            payload["stop"] = list(cfg.stop_sequences)
        payload.update(cfg.extra_options)
        return payload

    def generate_one(self, record: DatasetRecord) -> ResponseItem:
        payload = self.build_payload(record)
        if "seed" in payload:
            self.seed_forward_count += 1
        start = time.perf_counter()
        raw = self._request("POST", "/chat/completions", payload)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if not isinstance(raw, dict):
            raise ConfigurationError(f"Unexpected response format: {type(raw).__name__}")
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise OpenAICompatibleConnectionError("Response contained no choices")
        first = choices[0]
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = first.get("text")
        text = "" if content is None else str(content)

        token_usage: dict[str, int] | None = None
        usage = raw.get("usage")
        if isinstance(usage, dict):
            p = usage.get("prompt_tokens")
            c = usage.get("completion_tokens")
            if isinstance(p, int) and isinstance(c, int) and not isinstance(p, bool) and not isinstance(c, bool):
                total = usage.get("total_tokens")
                token_usage = {
                    "prompt_tokens": p,
                    "completion_tokens": c,
                    "total_tokens": total if isinstance(total, int) and not isinstance(total, bool) else p + c,
                }
        return ResponseItem(id=record.id, response=text, latency_ms=round(elapsed_ms, 3), token_usage=token_usage)

