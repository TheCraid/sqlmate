"""Minimal client for any OpenAI-compatible chat API: Groq, or a local Ollama server (/v1)."""

from __future__ import annotations

import os
import re
import time

import httpx

try:  # read GROQ_API_KEY from a .env file in the project folder, if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class APIError(RuntimeError):
    pass


class QuotaExhausted(APIError):
    """The provider's daily limit (or a very long rate-limit wait) was hit: stop now and resume later."""


def _retry_after(resp: httpx.Response) -> float:
    header = resp.headers.get("retry-after")
    if header:
        try:
            return float(header) + 0.5
        except ValueError:
            pass
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)(m?s)", resp.text)
    if m:
        secs = float(m.group(2)) / (1000 if m.group(3) == "ms" else 1)
        return int(m.group(1) or 0) * 60 + secs + 0.5
    return 5.0


class ChatClient:
    def __init__(self, base_url: str = "https://api.groq.com/openai/v1", api_key: str | None = None,
                 timeout_s: float = 120.0, max_retries: int = 10, max_wait_s: float = 90.0):
        key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY", "")
        if not key and "groq.com" in base_url:
            raise APIError("GROQ_API_KEY is not set. Put it in a .env file in the project folder.")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_s)
        self.max_retries, self.max_wait_s = max_retries, max_wait_s

    def chat(self, model: str, messages: list[dict], temperature: float = 0.0, json_mode: bool = False,
             max_tokens: int | None = None, reasoning_effort: str | None = None) -> tuple[str, dict]:
        """Return (reply text, {"input_tokens", "output_tokens", "latency_ms"}). Waits out short rate limits.

        Raises QuotaExhausted when the wait would be longer than max_wait_s (for example a daily token limit).
        """
        body: dict = {"model": model, "messages": messages, "temperature": temperature}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if max_tokens:
            body["max_tokens"] = max_tokens
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        start = time.perf_counter()
        resp = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.http.post("/chat/completions", json=body)
            except httpx.TransportError as exc:
                if attempt == self.max_retries:
                    raise APIError(f"{model}: network error: {exc}") from exc
                print(f"    network error ({exc.__class__.__name__}); retrying in 5 s")
                time.sleep(5)
                continue
            if resp.status_code == 429:
                wait = _retry_after(resp)
                if wait > self.max_wait_s or "per day" in resp.text.lower():
                    raise QuotaExhausted(f"{model}: limit reached, retry in about {wait / 60:.0f} min. "
                                         f"{resp.text[:200]}")
                if attempt < self.max_retries:
                    print(f"    rate limited; waiting {wait:.1f} s")
                    time.sleep(wait)
                    continue
            break
        if resp.status_code >= 400:
            raise APIError(f"{model}: HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        usage = data.get("usage") or {}
        return (data["choices"][0]["message"].get("content") or "",
                {"input_tokens": int(usage.get("prompt_tokens", 0)),
                 "output_tokens": int(usage.get("completion_tokens", 0)),
                 "latency_ms": round((time.perf_counter() - start) * 1000, 1)})
