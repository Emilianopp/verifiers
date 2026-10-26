"""`AgentConfig.segments`: an env-driven exchange runs as a chain of traces under a
token budget, joined by the model's own handoff summaries, against a recording
upstream whose usage grows with the conversation."""

import pytest

import verifiers.v1 as vf
from verifiers.v1.clients import EvalClientConfig
from verifiers.v1.configs.agent import SegmentsConfig


@pytest.fixture
async def recording_model():
    """A chat-completions upstream that answers "reply <n>" (n counts requests),
    reports 100 prompt tokens per message, and records every request body."""
    from aiohttp import web

    from verifiers.v1.dialects.chat import ChatDialect

    dialect = ChatDialect()
    requests: list[dict] = []

    async def answer(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        requests.append(body)
        raw = {
            "id": "chatcmpl-recorded",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "recorded"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"reply {len(requests)}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100 * len(body["messages"]),
                "completion_tokens": 10,
                "total_tokens": 100 * len(body["messages"]) + 10,
            },
        }
        if not dialect.streaming(body):
            return web.json_response(raw)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for event in dialect.stream_events(raw):
            await response.write(event)
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/{tail:.*}", answer)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/v1", requests
    finally:
        await runner.cleanup()


def _text(message: dict) -> str:
    content = message.get("content")
    return content if isinstance(content, str) else "".join(part.get("text", "") for part in content or [])


async def test_segments_switch_on_budget(recording_model, monkeypatch):
    from verifiers.v1.harnesses.null import NullHarnessConfig
    from verifiers.v1.runtimes import SubprocessConfig

    monkeypatch.setenv("RECORDED_API_KEY", "unused")
    base_url, requests = recording_model
    segments = SegmentsConfig(budget_tokens=700, summary_tokens=200)
    agent = vf.make_agent(
        vf.AgentConfig(
            harness=NullHarnessConfig(id="null"),
            runtime=SubprocessConfig(),
            model="recorded",
            sampling=vf.SamplingConfig(max_tokens=64),
            client=EvalClientConfig(base_url=base_url, api_key_var="RECORDED_API_KEY"),
            segments=segments,
        ),
    )
    task = vf.Task(vf.TaskData(idx=0, prompt=None, system_prompt="Play the game."))
    traces: list[vf.Trace] = []
    replies: list[str] = []
    async with agent.interaction(task, on_trace=lambda trace: traces.append(trace)) as interaction:
        for step in range(6):
            segment = await interaction.turn(f"observation {step}")
            assert not segment.terminated
            replies.append(segment.last_reply)
        last_trace = interaction.trace

    summary_requests = [i for i, body in enumerate(requests) if _text(body["messages"][-1]) == segments.summary_prompt]
    assert summary_requests, "the budget never triggered a summary"
    first = summary_requests[0]
    asked = requests[first]
    assert asked["max_tokens"] == 200  # the summary turn's own cap
    assert _text(asked["messages"][-2]).startswith("observation")  # the pending observation, in view
    # The next segment: the task's system prompt + the summary, opened by that observation.
    opened = requests[first + 1]["messages"]
    assert [m["role"] for m in opened] == ["system", "user"]
    assert _text(opened[0]).startswith("Play the game.")
    assert f"{segments.summary_header}\nreply {first + 1}" in _text(opened[0])
    assert _text(opened[1]) == _text(asked["messages"][-2])
    assert requests[first + 1]["max_tokens"] == 64
    # The caller never sees a summary: every reply it got answers an observation.
    assert f"reply {first + 1}" not in replies and len(replies) == 6
    # Each segment is its own trace, numbered.
    segment_ids = {trace.id: trace.info.get("segment") for trace in traces}
    assert len(segment_ids) >= 2 and last_trace.info["segment"] == len(segment_ids) - 1
