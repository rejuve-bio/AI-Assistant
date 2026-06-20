import contextvars
import os
import threading
import time
import uuid
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable


_current_tracker = contextvars.ContextVar("llm_usage_tracker", default=None)
_current_step = contextvars.ContextVar("llm_usage_step", default=None)


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def is_langsmith_enabled() -> bool:
    return bool(os.getenv("LANGSMITH_API_KEY")) and (
        _truthy_env("LANGSMITH_TRACING") or _truthy_env("LANGCHAIN_TRACING_V2")
    )


def _model_key(provider: str | None, model: str | None) -> str:
    raw = "_".join(part for part in (provider, model) if part)
    return "".join(ch if ch.isalnum() else "_" for ch in raw.upper())


def _price_from_env(
    provider: str | None,
    model: str | None,
) -> tuple[float | None, float | None]:
    key = _model_key(provider, model)
    input_price = os.getenv(f"LLM_COST_{key}_INPUT_PER_1M")
    output_price = os.getenv(f"LLM_COST_{key}_OUTPUT_PER_1M")
    try:
        input_value = float(input_price) if input_price else None
    except ValueError:
        input_value = None
    try:
        output_value = float(output_price) if output_price else None
    except ValueError:
        output_value = None
    return input_value, output_value


def estimate_tokens(text: Any) -> int:
    if text is None:
        return 0
    if not isinstance(text, str):
        text = str(text)
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4) if text else 0


def extract_usage_metadata(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is not None:
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if isinstance(usage, dict):
            return usage
        return {
            key: getattr(usage, key)
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "input_tokens",
                "output_tokens",
            )
            if getattr(usage, key, None) is not None
        }

    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        return usage

    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        for key in ("token_usage", "usage_metadata"):
            value = metadata.get(key)
            if isinstance(value, dict):
                return value
    return {}


def normalize_usage(
    usage_data: dict[str, Any] | None,
    *,
    prompt_text: str,
    completion_text: str,
) -> dict[str, Any]:
    usage_data = usage_data or {}
    prompt_tokens = (
        usage_data.get("input_tokens")
        or usage_data.get("prompt_tokens")
        or usage_data.get("input_token_count")
    )
    completion_tokens = (
        usage_data.get("output_tokens")
        or usage_data.get("completion_tokens")
        or usage_data.get("output_token_count")
    )
    total_tokens = usage_data.get("total_tokens") or usage_data.get(
        "total_token_count"
    )

    estimated = False
    if prompt_tokens is None:
        prompt_tokens = estimate_tokens(prompt_text)
        estimated = True
    if completion_tokens is None:
        completion_tokens = estimate_tokens(completion_text)
        estimated = True
    if total_tokens is None:
        total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)

    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
        "estimated": estimated,
    }


def wrap_openai_client(client):
    if not is_langsmith_enabled():
        return client, False
    try:
        from langsmith.wrappers import wrap_openai

        return wrap_openai(client), True
    except Exception:
        return client, False


class QueryUsageTracker:
    def __init__(
        self,
        *,
        user_id: str,
        query: str,
        metadata: dict[str, Any] | None = None,
    ):
        self.query_id = str(uuid.uuid4())
        self.user_id = user_id
        self.query = query
        self.metadata = metadata or {}
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": None,
        }

    def __deepcopy__(self, memo):
        # LangGraph may copy state values; this tracker is intentionally shared.
        return self

    def record_llm_call(
        self,
        *,
        provider: str | None,
        model: str | None,
        operation: str,
        usage: dict[str, Any],
        elapsed_ms: int,
        error: str | None = None,
    ) -> None:
        step = _current_step.get() or operation
        input_price, output_price = _price_from_env(provider, model)
        estimated_cost = None
        if input_price is not None and output_price is not None:
            estimated_cost = (
                usage["prompt_tokens"] * input_price
                + usage["completion_tokens"] * output_price
            ) / 1_000_000

        call = {
            "step": step,
            "operation": operation,
            "provider": provider,
            "model": model,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "estimated": usage["estimated"],
            "elapsed_ms": elapsed_ms,
        }
        if estimated_cost is not None:
            call["estimated_cost_usd"] = round(estimated_cost, 8)
        if error:
            call["error"] = error

        with self._lock:
            self.calls.append(call)
            self.totals["prompt_tokens"] += usage["prompt_tokens"]
            self.totals["completion_tokens"] += usage["completion_tokens"]
            self.totals["total_tokens"] += usage["total_tokens"]
            if estimated_cost is not None:
                current = self.totals["estimated_cost_usd"] or 0.0
                self.totals["estimated_cost_usd"] = round(
                    current + estimated_cost,
                    8,
                )

    def langchain_config(
        self,
        *,
        provider: str | None,
        model: str | None,
        operation: str,
    ) -> dict[str, Any]:
        step = _current_step.get() or operation
        return {
            "run_name": f"{step}:{operation}",
            "tags": [
                "rejuv-ai-assistant",
                f"step:{step}",
                f"provider:{provider or 'unknown'}",
            ],
            "metadata": self._run_metadata(provider, model, operation, step),
        }

    def langsmith_extra(
        self,
        *,
        provider: str | None,
        model: str | None,
        operation: str,
    ) -> dict[str, Any]:
        step = _current_step.get() or operation
        return {
            "name": f"{step}:{operation}",
            "tags": ["rejuv-ai-assistant", f"step:{step}"],
            "metadata": self._run_metadata(provider, model, operation, step),
        }

    def _run_metadata(self, provider, model, operation, step):
        return {
            "query_id": self.query_id,
            "user_id": self.user_id,
            "step": step,
            "operation": operation,
            "model_provider": provider,
            "model": model,
            **self.metadata,
        }

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "query_id": self.query_id,
                "langsmith": {
                    "enabled": is_langsmith_enabled(),
                    "project": os.getenv("LANGSMITH_PROJECT")
                    or os.getenv("LANGCHAIN_PROJECT"),
                },
                "totals": dict(self.totals),
                "llm_calls": [dict(call) for call in self.calls],
            }


@contextmanager
def start_query_tracking(
    *,
    user_id: str,
    query: str,
    metadata: dict[str, Any] | None = None,
):
    tracker = QueryUsageTracker(user_id=user_id, query=query, metadata=metadata)
    token = _current_tracker.set(tracker)
    try:
        yield tracker
    finally:
        _current_tracker.reset(token)


@contextmanager
def bind_usage_tracker(tracker):
    token = _current_tracker.set(tracker)
    try:
        yield
    finally:
        _current_tracker.reset(token)


@contextmanager
def llm_usage_step(step_name: str):
    token = _current_step.set(step_name)
    try:
        yield
    finally:
        _current_step.reset(token)


def current_usage_tracker() -> QueryUsageTracker | None:
    return _current_tracker.get()


def tracked_node(step_name: str, fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(*args, **kwargs):
        state = args[0] if args and isinstance(args[0], dict) else {}
        tracker = state.get("_usage_tracker") or current_usage_tracker()
        current_step = state.get("current_step") or {}
        resolved_name = step_name
        if step_name == "step_executor" and current_step:
            resolved_name = (
                f"step_{current_step.get('id', 0)}:"
                f"{current_step.get('agent', 'unknown')}"
            )
        with bind_usage_tracker(tracker), llm_usage_step(resolved_name):
            return fn(*args, **kwargs)

    return wrapper
