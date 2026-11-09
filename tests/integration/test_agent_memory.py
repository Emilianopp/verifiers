"""`AgentConfig.memory` through `Agent.interaction`: the system prompt explains the
tags, every observation leads with the [Memory] block, and the store survives a segment
switch (the summary request names it, the next segment opens on it)."""

import pytest

import verifiers.v1 as vf
from verifiers.v1.clients import EvalClientConfig
from verifiers.v1.configs.agent import MemoryConfig, SegmentsConfig
from verifiers.v1.memory import RESET, Memory


@pytest.fixture
async def writing_model():
    """Answers request n with `<write key="log">n</write> reply n`, 100 prompt tokens per
    message, and records every request body."""
    from aiohttp import web

    from verifiers.v1.dialects.chat import ChatDialect

    dialect = ChatDialect()
    requests: list[dict] = []

    async def answer(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        requests.append(body)
        n = len(requests)
        raw = {
            "id": "chatcmpl-recorded", "object": "chat.completion", "created": 0, "model": body.get("model", "recorded"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": f'<write key="log">{n}</write> reply {n}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100 * len(body["messages"]), "completion_tokens": 10, "total_tokens": 100 * len(body["messages"]) + 10},
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


async def test_memory_across_a_segment_switch(writing_model, monkeypatch):
    from verifiers.v1.harnesses.null import NullHarnessConfig
    from verifiers.v1.runtimes import SubprocessConfig

    monkeypatch.setenv("RECORDED_API_KEY", "unused")
    base_url, requests = writing_model
    segments, memory_config = SegmentsConfig(new_tokens=500, summary_tokens=200), MemoryConfig()
    agent = vf.make_agent(
        vf.AgentConfig(
            harness=NullHarnessConfig(id="null"), runtime=SubprocessConfig(), model="recorded",
            sampling=vf.SamplingConfig(max_tokens=64),
            client=EvalClientConfig(base_url=base_url, api_key_var="RECORDED_API_KEY"),
            segments=segments, memory=memory_config,
        ),
    )
    task = vf.Task(vf.TaskData(idx=0, prompt=None, system_prompt="Play the game."))
    store = Memory()
    async with agent.interaction(task, memory=store) as interaction:
        for step in range(6):
            assert not (await interaction.turn(f"observation {step}")).terminated

    assert _text(requests[0]["messages"][0]).startswith("Play the game.\n\nMEMORY\n")
    assert _text(requests[0]["messages"][-1]) == f"{Memory().index()}\n\nobservation 0\n\n{memory_config.turn_note}"
    assert _text(requests[1]["messages"][-1]).startswith('[Memory] log (1 tok)\nwrite "log": saved (1 tok)\n\nobservation 1')
    asked = next(i for i, body in enumerate(requests) if f"{memory_config.summary_note}\n\n{memory_config.turn_note}" in _text(body["messages"][-1]))
    opened = requests[asked + 1]["messages"]
    assert [m["role"] for m in opened] == ["system", "user"] and "MEMORY\n" in _text(opened[0])
    assert RESET in _text(opened[1]) and "write \"log\": saved" in _text(opened[1])  # the summary turn's write
    assert store.entries["log"] == "\n".join(str(n) for n in range(1, len(requests) + 1))
