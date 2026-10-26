"""Segments: one env-driven exchange as a chain of traces, each under a token
budget, joined by handoff summaries the model writes itself.

When the current trace reaches `budget_tokens - summary_tokens`, the next
`turn(message)` first asks the model, on that trace, for a summary of its
progress (with `message` in view, capped at `summary_tokens`), closes the trace,
and opens the next one: the task's system prompt plus the summary, opened by
`message`, whose reply is what the caller gets. The caller (an env's rollout
loop) keeps its own state, a game, across segments and never sees the switch.

Each segment is a full rollout with its own trace, limits and scoring; on a
KV-Streams server it is also its own session, so its positions start again from
zero and a run's length is no longer bounded by the model's position range. The
summary turn is the last turn of its segment's trace, so it is trained with it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING

from verifiers.v1.configs.agent import SegmentsConfig
from verifiers.v1.dialects import parse_message
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages, UserMessage

if TYPE_CHECKING:
    from verifiers.v1.agent import Interaction, Segment

OpenSegment = Callable[[Task], AbstractAsyncContextManager["Interaction"]]


def trace_tokens(trace: Trace) -> int:
    """The trace's chronological length: its message tokens (a KV-Streams trace's
    positions), else the provider-reported total when nodes carry no token ids."""
    return sum(len(node.token_ids) for node in trace.nodes) or trace.num_total_tokens


class SegmentedInteraction:
    """`Interaction.turn` and `trace` over a chain of segment interactions."""

    def __init__(self, config: SegmentsConfig, task: Task, open_segment: OpenSegment) -> None:
        self.config = config
        self._task = task
        self._open = open_segment
        self._stack = AsyncExitStack()
        self._current: Interaction | None = None
        self.segments = 0

    @property
    def trace(self) -> Trace:
        """The current segment's trace."""
        assert self._current is not None
        return self._current.trace

    async def turn(self, message: str | Messages | None = None, *, max_tokens: int | None = None) -> Segment:
        assert self._current is not None
        threshold = self.config.budget_tokens - self.config.summary_tokens
        if message is not None and trace_tokens(self.trace) >= threshold:
            await self._next_segment(message)
        return await self._current.turn(message, max_tokens=max_tokens)

    async def _start(self, task: Task) -> None:
        self._current = await self._stack.enter_async_context(self._open(task))
        self._current.trace.info["segment"] = self.segments
        self.segments += 1

    async def _next_segment(self, message: str | Messages) -> None:
        assert self._current is not None
        pending = [UserMessage(content=message)] if isinstance(message, str) else list(message)
        ask = [*(parse_message(m) if isinstance(m, dict) else m for m in pending), UserMessage(content=self.config.summary_prompt)]
        reply = await self._current.turn(ask, max_tokens=self.config.summary_tokens)
        summary = "" if reply.terminated else reply.last_reply
        await self._stack.aclose()  # the segment's rollout finishes: scoring and hooks run
        self._stack = AsyncExitStack()
        system = f"{self._task.data.system_prompt or ''}\n\n{self.config.summary_header}\n{summary}".lstrip()
        await self._start(self._task.with_data(system_prompt=system, prompt=None))


@asynccontextmanager
async def segmented(config: SegmentsConfig, task: Task, open_segment: OpenSegment) -> AsyncIterator[SegmentedInteraction]:
    interaction = SegmentedInteraction(config, task, open_segment)
    await interaction._start(task)
    try:
        yield interaction
    except BaseException as e:
        await interaction._stack.__aexit__(type(e), e, e.__traceback__)
        raise
    else:
        await interaction._stack.aclose()
