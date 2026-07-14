"""Python client SDK for the deployed Document Analyst (Part 3)."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator

import httpx


class AnalystClientError(Exception):
    def __init__(self, message: str, status_code: int | None = None, request_id: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


def _extract_content(data) -> str:
    """Extract the final answer text from either response shape the endpoint
    might return: a standard OpenAI `ChatCompletion` object, or — what this
    endpoint actually returns — a raw list wrapping the graph's own state
    dict. MLflow only auto-wraps a model's output into the OpenAI envelope
    when it conforms to a pure messages-only schema; our `AnalystState` has
    extra fields (`plan`, `step_results`, ...), so it's served as-is instead.
    """
    if isinstance(data, dict) and "choices" in data:
        return data["choices"][0]["message"]["content"]
    if isinstance(data, list) and data and isinstance(data[0], dict) and "messages" in data[0]:
        return data[0]["messages"][-1]["content"]
    raise AnalystClientError(f"Unrecognized response shape from endpoint: {data!r}"[:500])


class DocumentAnalystClient:
    def __init__(
        self,
        endpoint_name: str,
        host: str | None = None,
        token: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.endpoint_name = endpoint_name
        self.host = (host or os.environ.get("DATABRICKS_HOST", "")).rstrip("/")
        self.token = token or os.environ.get("DATABRICKS_TOKEN", "")
        if not self.host or not self.token:
            raise OSError(
                "DATABRICKS_HOST and DATABRICKS_TOKEN must be provided explicitly "
                "or set in the environment."
            )
        self.timeout = timeout
        self.max_retries = max_retries

        self._invocations_url = f"{self.host}/serving-endpoints/{self.endpoint_name}/invocations"
        self._status_url = f"{self.host}/api/2.0/serving-endpoints/{self.endpoint_name}"
        self._headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def _post_with_retry(self, payload: dict) -> httpx.Response:
        start = time.monotonic()
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self._invocations_url, headers=self._headers, json=payload)
            except httpx.TimeoutException as exc:
                elapsed = time.monotonic() - start
                raise TimeoutError(
                    f"Request to '{self.endpoint_name}' timed out after {elapsed:.1f}s"
                ) from exc

            if response.status_code in (429, 503) and attempt < self.max_retries:
                time.sleep(2**attempt)
                continue

            if response.status_code >= 400:
                raise AnalystClientError(
                    f"Endpoint '{self.endpoint_name}' returned {response.status_code}: {response.text}",
                    status_code=response.status_code,
                    request_id=response.headers.get("x-request-id"),
                )

            return response

        raise AssertionError("unreachable")  # loop always returns or raises above

    def ask(self, question: str) -> str:
        payload = {"messages": [{"role": "user", "content": question}]}
        response = self._post_with_retry(payload)
        return _extract_content(response.json())

    def ask_streaming(self, question: str) -> Iterator[str]:
        """Yield text chunks as they arrive.

        A models-from-code LangChain endpoint may not implement `predict_stream`.
        That can show up two ways: it silently ignores `stream: True` and
        returns a single complete JSON response (handled below via
        content-type), or — what this endpoint actually does — it rejects
        the request outright with a 400 `"This endpoint does not support
        streaming"` error. Detect both cases and fall back to yielding the
        full answer once, rather than assuming token-by-token deltas always
        arrive.
        """
        payload = {"messages": [{"role": "user", "content": question}], "stream": True}
        start = time.monotonic()
        try:
            with httpx.Client(timeout=self.timeout) as client, client.stream(
                "POST", self._invocations_url, headers=self._headers, json=payload
            ) as response:
                if response.status_code >= 400:
                    body = response.read().decode()
                    if response.status_code == 400 and "does not support streaming" in body:
                        yield self.ask(question)
                        return
                    raise AnalystClientError(
                        f"Endpoint '{self.endpoint_name}' returned {response.status_code}: {body}",
                        status_code=response.status_code,
                        request_id=response.headers.get("x-request-id"),
                    )

                content_type = response.headers.get("content-type", "")
                if "text/event-stream" not in content_type:
                    data = json.loads(response.read())
                    yield _extract_content(data)
                    return

                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[len("data:") :].strip()
                    if raw == "[DONE]":
                        break
                    chunk = json.loads(raw)
                    piece = chunk["choices"][0].get("delta", {}).get("content")
                    if piece:
                        yield piece
        except httpx.TimeoutException as exc:
            elapsed = time.monotonic() - start
            raise TimeoutError(
                f"Streaming request to '{self.endpoint_name}' timed out after {elapsed:.1f}s"
            ) from exc

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(self._status_url, headers=self._headers)
        except httpx.TimeoutException:
            return False
        if response.status_code != 200:
            return False
        state = response.json().get("state", {})
        return state.get("ready") == "READY"
