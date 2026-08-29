"""Shared utilities for Switchboard."""

import asyncio
import concurrent.futures
from typing import Any, Callable, Coroutine, Optional, TypeVar

_MAX_ERROR_BODY_LENGTH = 200

# Extra margin added to the thread-join timeout so the underlying HTTP
# timeout fires first with a meaningful error.
_SYNC_TIMEOUT_MARGIN = 10

T = TypeVar("T")


def summarize_error_body(response: Any) -> str:
    """Summarize an HTTP error response body for logs and exception messages.

    Prefers the structured error message when the body is JSON; otherwise
    falls back to the raw text. The result is collapsed to a single line and
    truncated so unbounded upstream content never propagates into logs or
    user-facing exceptions.

    Args:
        response: An httpx.Response-like object with .json() and .text

    Returns:
        A single-line summary, at most ~200 characters
    """
    message = ""
    try:
        data = response.json()
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                message = str(error.get("message", ""))
            elif isinstance(error, str):
                message = error
    except Exception:
        pass

    if not message:
        try:
            message = response.text or ""
        except Exception:
            message = "<unreadable response body>"

    # Collapse to a single line to prevent log injection via embedded newlines
    message = " ".join(message.split())

    if len(message) > _MAX_ERROR_BODY_LENGTH:
        message = message[:_MAX_ERROR_BODY_LENGTH] + "..."

    return message


def run_coroutine_sync(
    coro_factory: Callable[[], Coroutine[Any, Any, T]],
    timeout: Optional[float] = None,
) -> T:
    """Run a coroutine from synchronous code, event loop or not.

    If no event loop is running, uses asyncio.run directly. If called from
    within a running loop, runs the coroutine in a worker thread with its own
    loop. The coroutine is created inside the chosen branch (via the factory)
    so no un-awaited coroutine is left behind on error paths.

    Args:
        coro_factory: Zero-argument callable returning the coroutine to run
        timeout: Expected upper bound for the operation in seconds. The
            thread-join timeout gets a small margin on top so the operation's
            own timeout fires first.

    Returns:
        The coroutine's result
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, safe to use asyncio.run
        return asyncio.run(coro_factory())

    # Already in an event loop: run in a dedicated thread with its own loop
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future: concurrent.futures.Future[T] = executor.submit(
            asyncio.run, coro_factory()
        )
        join_timeout = (timeout + _SYNC_TIMEOUT_MARGIN) if timeout else None
        return future.result(timeout=join_timeout)
