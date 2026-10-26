"""`Interaction.turn(max_tokens=)` caps one turn's model calls, against a recording
upstream: the other turns keep the agent's own `sampling.max_tokens`."""

import pytest

import verifiers.v1 as vf
from verifiers.v1.clients import EvalClientConfig


@pytest.fixture
async def recording_model():
    """A chat-completions upstream that answers "OK" and records each request's
    `max_tokens` (or `max_completion_tokens`)."""
    from aiohttp import web

    from verifiers.v1.dialects.chat import ChatDialect

    dialect = ChatDialect()
    seen: list[int | None] = []

    async def answer(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        seen.append(body.get("max_tokens", body.get("max_completion_tokens")))
        raw = {
            "id": "chatcmpl-recorded",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "recorded"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "OK"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
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
        yield f"http://127.0.0.1:{port}/v1", seen
    finally:
        await runner.cleanup()


async def test_turn_max_tokens_caps_only_that_turn(recording_model, monkeypatch):
    from verifiers.v1.harnesses.null import NullHarnessConfig
    from verifiers.v1.runtimes import SubprocessConfig

    monkeypatch.setenv("RECORDED_API_KEY", "unused")

    base_url, seen = recording_model
    agent = vf.make_agent(
        vf.AgentConfig(
            harness=NullHarnessConfig(id="null"),
            runtime=SubprocessConfig(),
            model="recorded",
            sampling=vf.SamplingConfig(max_tokens=64),
            client=EvalClientConfig(base_url=base_url, api_key_var="RECORDED_API_KEY"),
        ),
    )
    task = vf.Task(vf.TaskData(idx=0, prompt=None, system_prompt="Answer briefly."))
    async with agent.interaction(task) as interaction:
        await interaction.turn("first")
        await interaction.turn("summarize", max_tokens=4096)
        await interaction.turn("third")
    assert interaction.trace.errors == []
    assert seen == [64, 4096, 64]
