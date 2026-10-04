"""API paths with a fake OpenAI-compatible server: rate limits, daily limits and resuming."""

import json
import re

import httpx
import pytest

from sqlmate import data_synthetic, evaluate
from sqlmate.llm_api import ChatClient, QuotaExhausted

DAILY = {"error": {"message": "Rate limit reached on tokens per day (TPD): Limit 200000. Please try again in 7m12s."}}


class FakeServer:
    def __init__(self, fail_after: int | None = None):
        self.calls, self.fail_after, self.n = 0, fail_after, 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            return httpx.Response(429, json=DAILY)
        body = json.loads(request.content)
        text = body["messages"][-1]["content"]
        if body.get("response_format"):
            tables = re.findall(r"CREATE TABLE (\w+)", text)
            items = []
            for _ in range(5):
                self.n += 1
                items.append({"question": f"Unusual metric number {self.n} for table {tables[0]} please",
                              "sql": f"SELECT COUNT(*) * 1000 + {self.n} AS v FROM {tables[0]}"})
            content = json.dumps({"items": items})
        else:
            content = "SELECT 1"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 5}})


def fake_client(server: FakeServer) -> ChatClient:
    c = ChatClient("http://fake/v1", api_key="x")
    c.http = httpx.Client(base_url="http://fake/v1", transport=httpx.MockTransport(server))
    return c


def test_short_rate_limit_is_waited_out(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    replies = iter([httpx.Response(429, headers={"retry-after": "2"}, json={}),
                    httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})])
    c = ChatClient("http://fake/v1", api_key="x")
    c.http = httpx.Client(base_url="http://fake/v1", transport=httpx.MockTransport(lambda r: next(replies)))
    assert c.chat("m", [{"role": "user", "content": "hi"}])[0] == "ok"


def test_daily_limit_raises():
    with pytest.raises(QuotaExhausted):
        fake_client(FakeServer(fail_after=0)).chat("m", [{"role": "user", "content": "hi"}])


def test_missing_groq_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(Exception, match="GROQ_API_KEY"):
        ChatClient()


def run_synthetic(monkeypatch, tmp_path, server, per_dataset):
    monkeypatch.setattr(data_synthetic, "DATA", tmp_path)
    monkeypatch.setattr(data_synthetic, "ChatClient", lambda base_url: fake_client(server))
    monkeypatch.setattr("sys.argv", ["x", "--per-dataset", str(per_dataset), "--batch", "5"])
    data_synthetic.main()
    lines = (tmp_path / "synthetic_train.jsonl").read_text().splitlines()
    return [json.loads(x) for x in lines]


def test_synthetic_stops_on_daily_limit_and_resumes(monkeypatch, tmp_path):
    rows = run_synthetic(monkeypatch, tmp_path, FakeServer(fail_after=4), per_dataset=12)
    assert len(rows) == 12 + 5  # store done in 3 calls (5 + 5 + 2), food got one call, then the limit hit
    assert {r["dataset"] for r in rows} == {"store", "food"}
    server = FakeServer()
    server.n = 1000
    rows = run_synthetic(monkeypatch, tmp_path, server, per_dataset=12)
    counts = {d: sum(r["dataset"] == d for r in rows) for d in ("store", "food", "hr")}
    assert counts == {"store": 12, "food": 12, "hr": 12}
    assert len({r["completion"][0]["content"] for r in rows}) == 36


def test_evaluate_resumes_after_daily_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate, "RESULTS", tmp_path)
    server = FakeServer(fail_after=7)
    monkeypatch.setattr(evaluate.APIBackend, "__init__", lambda self, *a: (
        setattr(self, "client", fake_client(server)), setattr(self, "model", "m"),
        setattr(self, "sleep", 0), setattr(self, "reasoning_effort", None))[0])
    monkeypatch.setattr("sys.argv", ["x", "--backend", "api", "--model", "m", "--label", "Fake", "--sets", "datachat"])
    with pytest.raises(SystemExit):
        evaluate.main()
    assert len((tmp_path / ".partial-fake.jsonl").read_text().splitlines()) == 7
    server.fail_after = None
    evaluate.main()
    result = json.loads((tmp_path / "fake.json").read_text())
    assert result["sets"]["datachat"]["n"] == 60
    assert server.calls == 7 + 1 + 53
    assert not (tmp_path / ".partial-fake.jsonl").exists()
