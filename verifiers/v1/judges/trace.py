"""Built-in trace judge: one LLM call over a rendered trace, its score added to the reward.

For judging *behaviour* rather than an answer: the judge reads what the agent saw and did
(reasoning included by default) and returns a score from 0 to `scale`, recorded as a
reward in [0, 1] under the judge's name with the config's `weight`, beside the task's own
rewards. Plug it with no code:

    [[env.taskset.task.judges]]
    id = "trace"
    weight = 0.1
    model = "gemini-3.8-flash"
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key_var = "GEMINI_API_KEY"
    criteria = "Does the agent stop repeating actions that failed? ..."

or subclass it and override what differs — usually `render` (what the judge sees) and
`prompt`/`criteria` (what it is asked); `judge_trace` is the whole call and may return a
mapping of named scores instead of one:

    class MyJudgeConfig(vf.TraceJudgeConfig):
        id: vf.ID = "my_env.judge"

    class MyJudge(vf.TraceJudge[MyJudgeConfig]):
        def render(self, task, trace) -> str:
            ...

Each segment of a segmented interaction is its own trace, so it is judged on its own.
`on_error = "skip"` keeps a judge outage from failing a long episode: the failure is
recorded as the `<name>/error` metric and no judge reward is added.
"""

import json
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, Field
from typing_extensions import TypeVar

from verifiers.v1.configs.judge import JudgeConfig
from verifiers.v1.judge import Judge, JudgeResponse
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import ID, AssistantMessage, SystemMessage, ToolMessage, content_text

logger = logging.getLogger(__name__)

TRACE_PROMPT = (Path(__file__).resolve().parent / "trace.txt").read_text(encoding="utf-8")

CRITERIA = (
    "Judge the process, not only the outcome. A good agent works toward the task's goal, "
    "uses the feedback it gets, stops repeating actions that did not work, and keeps what "
    "it says it believes consistent with what it observed. A bad one loops, ignores "
    "feedback, or acts on beliefs the trace contradicts."
)

JSON_SUFFIX = (
    "\n\nRespond with ONLY a JSON object and nothing else, in exactly this shape:\n"
    '{{"reason": "<2-4 sentences citing specific moments in the trace>", "score": <integer 0 to {scale}>}}\n'
    "Write the reason first; the score follows from it."
)


class TraceJudgeConfig(JudgeConfig):
    id: ID = "trace"
    criteria: str = CRITERIA
    """What to judge: fills the prompt's `{criteria}`."""
    scale: int = Field(10, ge=1)
    """The judge scores 0..scale; the reward is score / scale."""
    reasoning: bool = True
    """Show the agent's reasoning in the default `render`."""
    max_chars: int = Field(400_000, gt=0)
    """The rendered trace is cut to its last `max_chars` characters."""
    on_error: Literal["raise", "skip"] = "raise"
    """A failed call or unparseable verdict: `raise` fails the rollout (a judge failure is
    never scored against the model); `skip` records `<name>/error` = 1 and no reward."""


class TraceVerdict(BaseModel):
    reason: str
    score: float


TraceConfigT = TypeVar("TraceConfigT", bound=TraceJudgeConfig, default=TraceJudgeConfig)


def fence_for(*texts: str) -> str:
    """A code fence longer than any backtick run in `texts`, so none can close it early."""
    longest = max((len(run) for text in texts for run in re.findall(r"`+", text)), default=2)
    return "`" * (longest + 1)


def json_object_with(text: str, key: str) -> dict | None:
    """The first balanced JSON object in `text` that has `key` (skipping prose and an echoed
    format example)."""
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and key in obj:
            return obj
        idx = text.find("{", idx + 1)
    return None


class TraceJudge(Judge[TraceVerdict, TraceConfigT]):
    prompt = TRACE_PROMPT

    def task_text(self, task: TaskData, trace: Trace) -> str:
        """What the agent was asked: its system prompt, then the task prompt."""
        branches = trace.branches
        system = [
            content_text(m.content)
            for m in (branches[-1].messages if branches else [])
            if isinstance(m, SystemMessage)
        ]
        return "\n\n".join(text for text in [*system[:1], task.prompt_text] if text)

    def render(self, task: TaskData, trace: Trace) -> str:
        """What the agent saw and did, as text: every non-system message of the last
        branch, with reasoning (`config.reasoning`) and tool calls."""
        branches = trace.branches
        blocks: list[str] = []
        for message in branches[-1].messages if branches else []:
            if isinstance(message, SystemMessage):
                continue
            lines = [f"[{message.role}]"]
            if isinstance(message, AssistantMessage):
                if self.config.reasoning and message.reasoning_content:
                    lines.append(f"<reasoning>\n{message.reasoning_content}\n</reasoning>")
                if message.content:
                    lines.append(message.content)
                lines.extend(f"[tool_call {call.name}({call.arguments})]" for call in message.tool_calls or [])
            else:
                if isinstance(message, ToolMessage) and message.name:
                    lines[0] = f"[{message.role} {message.name}]"
                if text := content_text(message.content):
                    lines.append(text)
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def fields(self, task: TaskData, trace: Trace) -> dict[str, Any]:
        """The prompt's placeholders."""
        rendered = self.render(task, trace)
        if len(rendered) > self.config.max_chars:
            cut = len(rendered) - self.config.max_chars
            rendered = f"[... {cut:,} earlier characters cut ...]\n{rendered[cut:]}"
        text = self.task_text(task, trace)
        return {
            "task": text,
            "trace": rendered,
            "criteria": self.config.criteria,
            "scale": self.config.scale,
            "fence": fence_for(text, rendered),
        }

    def parse(self, response: JudgeResponse[TraceVerdict]) -> TraceVerdict:
        obj = json_object_with(response.text, "score")
        if obj is None:
            raise ValueError(f"judge returned no score JSON object: {response.text!r}")
        verdict = TraceVerdict.model_validate(obj)
        if not 0 <= verdict.score <= self.config.scale:
            raise ValueError(f"judge score {verdict.score} is outside 0..{self.config.scale}")
        return verdict

    async def judge_trace(self, task: TaskData, trace: Trace) -> float | Mapping[str, float]:
        """Judge one trace: build the prompt, call the model, parse the verdict."""
        messages = cast(str, self.build_messages(**self.fields(task, trace)))
        messages += JSON_SUFFIX.format(scale=self.config.scale)
        result = await self.complete(messages, trace=trace, parse=self.parse)
        return cast(TraceVerdict, result.parsed).score / self.config.scale

    async def score(self, task: TaskData, trace: Trace) -> float | Mapping[str, float]:
        try:
            return await self.judge_trace(task, trace)
        except Exception as e:
            if self.config.on_error == "raise":
                raise
            logger.warning("judge %s skipped: %s", self.reward_name, e)
            trace.record_metric(f"{self.reward_name}/error", 1.0)
            return {}  # no reward recorded under this judge's name


__all__ = ["TraceJudge", "TraceJudgeConfig", "TraceVerdict"]
