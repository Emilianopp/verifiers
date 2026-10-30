"""Segments: one env-driven exchange as a chain of traces, each under a budget of
new tokens, joined by handoff summaries the model writes itself.

When the current segment has added `new_tokens - summary_tokens` past its opening
system prompt (`segment_new_tokens`), the next `turn(message)` first asks the model,
on that trace, for a summary of its progress (with `message` in view, capped at
`summary_tokens`), closes the trace, and opens the next one: the task's system prompt
plus the summary, opened by `message`, whose reply is what the caller gets. The
inherited summary is not counted, so every segment gets the same budget. The caller (an env's rollout
loop) keeps its own state, a game, across segments and never sees the switch.

Each segment is a full rollout with its own trace, limits and scoring; on a
KV-Streams server it is also its own session, so its positions start again from
zero and a run's length is no longer bounded by the model's position range. The
summary turn is the last turn of its segment's trace, so it is trained with it.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING

from verifiers.v1.configs.agent import SegmentsConfig
from verifiers.v1.dialects import parse_message
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.types import AssistantMessage, Messages, UserMessage

if TYPE_CHECKING:
    from verifiers.v1.agent import Interaction, Segment

OpenSegment = Callable[[Task], AbstractAsyncContextManager["Interaction"]]
_SUMMARY = re.compile(r"<summary>(.*?)(?:</summary>|$)", re.DOTALL)


def segment_new_tokens(trace: Trace) -> int:
    """Tokens the segment has added past its opening system prompt, which carries the
    previous segment's summary: every later message's tokens, evicted ones included (a
    KV-Streams trace's positions past the system prompt). When nodes carry no token ids,
    the provider-reported total past the first call's prompt, which also leaves out the
    segment's opening message."""
    if any(node.token_ids for node in trace.nodes):
        return sum(len(node.token_ids) for node in trace.nodes if node.message.role != "system")
    opening = next((call.usage.input_tokens for call in trace.calls if call.usage is not None), 0)
    return max(0, trace.num_total_tokens - opening)


def _summary_text(reply: str) -> str:
    """The text inside `<summary>...</summary>` (to the end of the reply when the closing
    tag is missing), else the whole reply."""
    match = _SUMMARY.search(reply)
    return (match.group(1) if match else reply).strip()


def _segment_summary(segment: Segment) -> str:
    """The summary a summary turn wrote: from its reply, else from its reasoning. A
    thinking model often writes the summary in its think block and stops there, leaving
    an empty reply; that would hand the next segment nothing."""
    summary = _summary_text(segment.last_reply)
    if summary or segment.root_reply is not None:
        return summary
    for message in reversed(segment.messages):
        if isinstance(message, AssistantMessage):
            return _summary_text(message.reasoning_content or "")
    return ""


class SegmentedInteraction:
    """`Interaction.turn` and `trace` over a chain of segment interactions."""

    def __init__(self, config: SegmentsConfig, task: Task, open_segment: OpenSegment) -> None:
        # `task` keeps its own system prompt: every segment's summary is added to it.
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
        threshold = self.config.new_tokens - self.config.summary_tokens
        if message is not None and segment_new_tokens(self.trace) >= threshold:
            await self._next_segment(message)
        return await self._current.turn(message, max_tokens=max_tokens)

    async def _start(self, task: Task) -> None:
        self._current = await self._stack.enter_async_context(self._open(task))
        self._current.trace.info["segment"] = self.segments
        self.segments += 1

    def _ask_for_summary(self, message: str | Messages) -> str | Messages:
        """`message` with the summary request appended to its last user message: the
        turn keeps the shape of every other turn (one new user message), so the train
        client renders it incrementally on the sampled tokens. A second user message
        would make it re-render the history from text, which does not always round-trip
        and would force a KV-Streams session to reopen."""
        request = self.config.summary_prompt
        if isinstance(message, str):
            return f"{message}\n\n{request}"
        messages = [parse_message(m) if isinstance(m, dict) else m for m in message]
        last = messages[-1] if messages else None
        if isinstance(last, UserMessage) and isinstance(last.content, str):
            return [*messages[:-1], last.model_copy(update={"content": f"{last.content}\n\n{request}"})]
        return [*messages, UserMessage(content=request)]

    async def summarize(self, message: str | Messages) -> str:
        """Ask for a handoff summary now, on the current segment (as a segment switch
        does, with `message` in view), and return its text ("" if the turn did not run).
        For a run that stops here and resumes later from its own saved state: the next
        run opens with `segmented(..., summary=...)`."""
        assert self._current is not None
        reply = await self._current.turn(self._ask_for_summary(message), max_tokens=self.config.summary_tokens)
        return "" if reply.terminated else _segment_summary(reply)

    def _with_summary(self, summary: str) -> Task:
        system = f"{self._task.data.system_prompt or ''}\n\n{self.config.summary_header}\n{summary}".lstrip()
        return self._task.with_data(system_prompt=system, prompt=None)

    async def _next_segment(self, message: str | Messages) -> None:
        assert self._current is not None
        summary = await self.summarize(message)
        await self._stack.aclose()  # the segment's rollout finishes: scoring and hooks run
        self._stack = AsyncExitStack()
        await self._start(self._with_summary(summary))


@asynccontextmanager
async def segmented(
    config: SegmentsConfig, task: Task, open_segment: OpenSegment, summary: str | None = None
) -> AsyncIterator[SegmentedInteraction]:
    """`summary`: a handoff summary for the first segment (a run resumed from a saved
    state, see `SegmentedInteraction.summarize`), placed as a segment switch places it."""
    interaction = SegmentedInteraction(config, task, open_segment)
    await interaction._start(task if summary is None else interaction._with_summary(summary))
    try:
        yield interaction
    except BaseException as e:
        await interaction._stack.__aexit__(type(e), e, e.__traceback__)
        raise
    else:
        await interaction._stack.aclose()
