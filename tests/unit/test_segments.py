"""Segments resumed from a saved state: a handoff summary opens the first segment, and
`summarize` asks for one without switching segments."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1.configs.agent import SegmentsConfig
from verifiers.v1.agent import Segment as Segment_
from verifiers.v1.segments import segmented

CONFIG = SegmentsConfig(new_tokens=100, summary_tokens=10, summary_header="HEADER")


class FakeSegments:
    """Opens fake segment interactions; every reply is `replies.pop(0)` ("content|think:reasoning"),
    and a turn adds `tokens` to its segment's count."""

    def __init__(self, replies: list[str], tokens: int = 0):
        self.replies = replies
        self.tokens = tokens
        self.systems: list[str] = []
        self.turns: list[tuple[str, int | None]] = []

    @asynccontextmanager
    async def open(self, task):
        self.systems.append(task.data.system_prompt)
        trace = SimpleNamespace(nodes=[], calls=[], num_total_tokens=0, info={})
        fake = self

        class Segment:
            async def turn(self, message, max_tokens=None):
                fake.turns.append((message, max_tokens))
                trace.num_total_tokens += fake.tokens
                reply = fake.replies.pop(0)
                content, _, reasoning = reply.partition("|think:")
                return Segment_(
                    messages=[vf.AssistantMessage(content=content or None, reasoning_content=reasoning or None)]
                )

        interaction = Segment()
        interaction.trace = trace
        yield interaction


def _task():
    return vf.Task(vf.TaskData(idx=0, prompt=None, system_prompt="sys"))


@pytest.mark.asyncio
async def test_a_resumed_run_opens_with_its_summary_and_does_not_stack_it():
    fake = FakeSegments(["<action>DO</action>", "<summary>new</summary>", "<action>NOOP</action>"], tokens=95)
    async with segmented(CONFIG, _task(), fake.open, summary="old") as interaction:
        await interaction.turn("obs 1")
        await interaction.turn("obs 2")  # past new_tokens - summary_tokens: switch first
    assert fake.systems == ["sys\n\nHEADER\nold", "sys\n\nHEADER\nnew"]


@pytest.mark.asyncio
async def test_summarize_asks_on_the_current_segment():
    fake = FakeSegments(["<action>DO</action>", "notes <summary>where I am</summary>"])
    async with segmented(CONFIG, _task(), fake.open) as interaction:
        await interaction.turn("obs 1")
        assert await interaction.summarize("obs 2") == "where I am"
    assert fake.systems == ["sys"]
    message, max_tokens = fake.turns[-1]
    assert message.startswith("obs 2\n\n") and message.endswith(CONFIG.summary_prompt)
    assert max_tokens == 10


@pytest.mark.asyncio
async def test_a_summary_written_in_the_think_block_is_kept():
    fake = FakeSegments(["<action>DO</action>", "|think:I should note <summary>wood at (3,4)</summary>"])
    async with segmented(CONFIG, _task(), fake.open) as interaction:
        await interaction.turn("obs 1")
        assert await interaction.summarize("obs 2") == "wood at (3,4)"
