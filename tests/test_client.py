"""Offline tests for the client SDK (Part 3) using a mocked HTTP transport.

Run:  uv run pytest -q
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import pytest

from client.sdk import AnalystClientError, DocumentAnalystClient


def _client(handler, **kwargs) -> DocumentAnalystClient:
    client = DocumentAnalystClient(
        endpoint_name="test-endpoint",
        host="https://example.databricks.com",
        token="dapi-test",
        max_retries=kwargs.pop("max_retries", 2),
        **kwargs,
    )
    transport = httpx.MockTransport(handler)
    _patch_transport(client, transport)
    return client


def _patch_transport(client: DocumentAnalystClient, transport: httpx.MockTransport) -> None:
    """Monkeypatch httpx.Client construction to use the mock transport."""
    import client.sdk as sdk_module

    original_client_cls = httpx.Client

    def _client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client_cls(*args, **kwargs)

    sdk_module.httpx.Client = _client_factory


@pytest.fixture(autouse=True)
def _restore_httpx_client():
    original = httpx.Client
    yield
    import client.sdk as sdk_module

    sdk_module.httpx.Client = original


def test_ask_returns_content_on_success():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "42"}}]})

    c = _client(handler)
    assert c.ask("what is the answer?") == "42"


def test_ask_parses_raw_state_list_shape():
    """This is what the actual deployed endpoint returns: MLflow only
    auto-wraps output into an OpenAI ChatCompletion envelope when the
    model's schema is pure messages-only; our AnalystState has extra
    fields, so it's served as a raw list wrapping the state dict instead.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "messages": [
                        {"content": "question", "type": "human"},
                        {"content": "the real answer", "type": "ai"},
                    ],
                    "plan": ["step 1"],
                    "final_answer": "the real answer",
                }
            ],
        )

    c = _client(handler)
    assert c.ask("hi") == "the real answer"


def test_ask_retries_on_429_then_succeeds():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    c = _client(handler, max_retries=2)
    assert c.ask("hi") == "ok"
    assert calls["count"] == 2


def test_ask_raises_analyst_client_error_on_4xx():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found", headers={"x-request-id": "req-123"})

    c = _client(handler, max_retries=0)
    with pytest.raises(AnalystClientError) as exc_info:
        c.ask("hi")
    assert exc_info.value.status_code == 404
    assert exc_info.value.request_id == "req-123"


def test_ask_raises_after_exhausting_retries_on_503():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="scaling up")

    c = _client(handler, max_retries=2)
    with pytest.raises(AnalystClientError) as exc_info:
        c.ask("hi")
    assert exc_info.value.status_code == 503


def test_ask_timeout_raises_timeout_error_with_elapsed_time():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("simulated timeout")

    c = _client(handler, max_retries=0, timeout=0.01)
    with pytest.raises(TimeoutError, match=r"timed out after \d"):
        c.ask("hi")


def test_health_check_true_when_ready():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"state": {"ready": "READY"}})

    c = _client(handler)
    assert c.health_check() is True


def test_health_check_false_when_not_ready():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"state": {"ready": "NOT_READY"}})

    c = _client(handler)
    assert c.health_check() is False


def test_ask_streaming_falls_back_to_single_chunk_for_non_sse_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "full answer"}}]},
            headers={"content-type": "application/json"},
        )

    c = _client(handler)
    chunks = list(c.ask_streaming("hi"))
    assert chunks == ["full answer"]


def test_ask_streaming_falls_back_when_endpoint_rejects_streaming():
    """The real deployed endpoint rejects `stream: True` outright with a 400
    rather than silently ignoring it — ask_streaming() must still degrade
    gracefully to a single non-streaming answer in that case."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if b'"stream": true' in request.content or b'"stream":true' in request.content:
            return httpx.Response(
                400,
                json={"error_code": "BAD_REQUEST", "message": "This endpoint does not support streaming."},
            )
        return httpx.Response(200, json=[{"messages": [{"content": "fallback answer"}]}])

    c = _client(handler)
    chunks = list(c.ask_streaming("hi"))
    assert chunks == ["fallback answer"]


def test_ask_streaming_parses_sse_deltas():
    sse_body = (
        'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
        'data: {"choices": [{"delta": {"content": " world"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse_body.encode(),
            headers={"content-type": "text/event-stream"},
        )

    c = _client(handler)
    chunks = list(c.ask_streaming("hi"))
    assert chunks == ["Hello", " world"]
