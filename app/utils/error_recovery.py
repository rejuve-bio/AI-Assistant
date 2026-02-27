"""
Error recovery utilities for LangGraph agent nodes.

Provides a `safe_agent_node` wrapper that adds:
- Configurable retry with exponential backoff
- Per-agent error tracking in AgentState
- Graceful degradation (returns error response instead of crashing)
- Rich logging for retry attempts and failures
"""

import time
import logging
import functools
from typing import Callable, Dict, Any, Optional

logger = logging.getLogger(__name__)


def safe_agent_node(
    agent_fn: Callable,
    agent_name: str,
    max_retries: int = 2,
    base_delay: float = 1.0,
    response_key: Optional[str] = None,
) -> Callable:
    """
    Wrap a LangGraph agent node function with retry and error recovery.

    This decorator ensures that:
    1. Transient failures are retried with exponential backoff
    2. Permanent failures are caught and recorded in state['agent_errors']
    3. The workflow never crashes due to a single agent failure
    4. Failed agents return a structured error response

    Args:
        agent_fn: The original agent node function (takes state, returns dict)
        agent_name: Human-readable name for logging
        max_retries: Maximum retry attempts (default: 2, so 3 total attempts)
        base_delay: Base delay in seconds between retries (doubles each retry)
        response_key: State key for this agent's response (e.g., 'rag_response')

    Returns:
        Wrapped function with the same signature

    Usage in _create_workflow():
        workflow.add_node("rag_agent", safe_agent_node(
            self.agents.rag_agent, "RAG Agent", response_key="rag_response"
        ))
    """

    @functools.wraps(agent_fn)
    def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.info(
                        f"🔄 Retrying {agent_name} (attempt {attempt + 1}/{max_retries + 1}) "
                        f"after {delay:.1f}s delay..."
                    )
                    # Log retry with RichLogger if available
                    try:
                        from app.utils import RichLogger
                        RichLogger.log_error(
                            f"{agent_name} Retry",
                            f"Attempt {attempt + 1}/{max_retries + 1} — "
                            f"Previous error: {last_error}"
                        )
                    except ImportError:
                        pass

                    time.sleep(delay)

                # Call the actual agent function
                result = agent_fn(state)

                # Validate result
                if result is None:
                    raise ValueError(f"{agent_name} returned None")

                # Check if agent reported its own error
                if result.get("error") and not result.get(response_key):
                    agent_error = result.get("error", "Unknown error")
                    if attempt < max_retries:
                        logger.warning(
                            f"{agent_name} reported error: {agent_error}, retrying..."
                        )
                        last_error = agent_error
                        continue

                # Success — return the result
                if attempt > 0:
                    logger.info(
                        f"✅ {agent_name} succeeded on attempt {attempt + 1}"
                    )
                return result

            except Exception as e:
                last_error = str(e)
                logger.error(
                    f"❌ {agent_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )

                if attempt == max_retries:
                    # Final failure — return graceful error response
                    logger.error(
                        f"🚫 {agent_name} permanently failed after "
                        f"{max_retries + 1} attempts. Last error: {last_error}"
                    )
                    break

        # Build error response that won't crash the workflow
        error_response = {
            "agents_completed": [agent_name.lower().replace(" ", "_")],
            "error": f"{agent_name} failed: {last_error}",
            "agent_errors": {agent_name: last_error},
        }

        # Set the agent's response key to an error marker
        if response_key:
            error_response[response_key] = {
                "text": f"[{agent_name} encountered an error: {last_error}]",
                "json_format": None,
                "source": agent_name,
                "failed": True,
            }

        return error_response

    # Preserve original function metadata for LangGraph introspection
    wrapper.__name__ = agent_fn.__name__
    wrapper.__qualname__ = agent_fn.__qualname__
    return wrapper
