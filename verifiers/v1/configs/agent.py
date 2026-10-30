"""One env agent's config: who plays the seat, and its per-run caps."""

from typing import Any

from pydantic import BaseModel, Field, SerializeAsAny, field_serializer, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.clients import ClientConfig
from verifiers.v1.configs.harness import HarnessConfig, WireHarnessConfig
from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.runtimes import PrimeConfig, RuntimeConfig
from verifiers.v1.types import SamplingConfig
from verifiers.v1.utils.generic import merge_defaults


class TimeoutConfig(BaseConfig):
    """Timeout (in seconds) for different phases of an agent's run. Each unset
    stage takes the task's own timeout for it, else runs with no limit."""

    # one shared budget: task setup + provisioning
    setup: float | None = Field(None, gt=0)
    """Timeout (in seconds) for task/harness setup through session preparation."""
    rollout: float | None = Field(None, gt=0)
    """Timeout (in seconds) for the agent's solve attempt."""
    finalize: float | None = Field(None, gt=0)
    """Timeout (in seconds) for the task + harness finalize hooks."""
    scoring: float | None = Field(None, gt=0)
    """Timeout (in seconds) for the task + harness metrics + scoring hooks."""


class SegmentsConfig(BaseConfig):
    """Run an env-driven exchange as a chain of traces (segments), each under a
    token budget, joined by handoff summaries the model writes itself. See
    `verifiers.v1.segments`."""

    new_tokens: int = Field(gt=0)
    """New tokens a segment may add: every message after its opening system prompt
    (observations, replies, its own closing summary), evicted ones included. The system
    prompt carries the previous segment's summary and is not counted, so every segment
    gets the same budget whatever that summary's length. The switch to the next segment
    starts once the segment has added `new_tokens - summary_tokens`; its trace is then
    at most the system prompt (with up to `summary_tokens` of inherited summary) plus
    `new_tokens`, give or take the turn that crosses the threshold."""
    summary_tokens: int = Field(gt=0)
    """Cap on the handoff summary's length, and the room kept for writing it."""
    summary_prompt: str = (
        "Your context is about to be reset, and this reply is not a game turn. Write a "
        "handoff summary of your progress between <summary> and </summary>: the goal, "
        "what you have achieved, what you know about the world and where things are, "
        "what failed, and your plan. It is all you will keep of this conversation; the "
        "observation above will be shown to you again after the reset. Do not write an "
        "action."
    )
    """Appended to the pending observation to ask for the summary. The text inside
    `<summary>...</summary>` is kept, else the whole reply."""
    summary_header: str = "HANDOFF SUMMARY (written by you before your context was reset):"
    """Heads the summary in the next segment's system prompt."""

    @model_validator(mode="after")
    def _summary_fits(self):
        if self.summary_tokens >= self.new_tokens:
            raise ValueError("segments.summary_tokens must be below segments.new_tokens")
        return self


class MemoryConfig(BaseConfig):
    """A memory the model keeps itself across an `interaction()`, segment switches
    included: named text entries it edits with tags in its replies. See
    `verifiers.v1.memory`."""

    max_key_tokens: int = Field(4096, gt=0)
    """Cap on one entry's length, in tokens estimated as characters / 4: an edit that
    would leave an entry above it and longer than before is not saved."""
    prompt: str = (
        "MEMORY\n"
        "You have a memory that is kept for the whole game, across context resets: named "
        "entries (keys) of text that you organize however you like. Edit it with tags in your "
        "reply, after your reasoning and before your action tags (tags inside your reasoning "
        "are ignored):\n"
        '  <write key="NAME">text</write> adds text to the end of an entry (creates it if new)\n'
        '  <rewrite key="NAME">text</rewrite> replaces the whole entry\n'
        '  <replace key="NAME"><old>exact text</old><new>new text</new></replace> replaces every '
        "occurrence of the exact text\n"
        '  <delete key="NAME">exact text</delete> removes that text; <delete key="NAME"/> removes '
        "the entry\n"
        '  <read key="NAME"/> shows the entry\'s full text at the top of your next observation\n'
        "Memory tags do not use up a turn. Every observation starts with a [Memory] line listing "
        "your keys and their sizes, then the results of your last edits. An entry holds at most "
        "{max_key_tokens} tokens: once it is full, an edit that makes it longer is not saved; "
        "free space with rewrite or delete. After a context reset, start by reading the entries "
        "you need."
    )
    """Added to the system prompt; `{max_key_tokens}` is filled in."""
    summary_note: str = (
        "Your memory is kept across the reset (its keys are listed at the top of this "
        "observation), and you may edit it in this reply too. In the summary, say which memory "
        "entries to read first after the reset."
    )
    """Appended to the segments' summary request."""


class AgentConfig(BaseConfig):
    harness: SerializeAsAny[HarnessConfig] | None = None
    """The agent's program (None = the taskset's default harness)."""
    runtime: RuntimeConfig = PrimeConfig()
    """Runtime for the harness program — the policy each run provisions its box
    from; tool servers choose their placement separately."""

    model: str | None = None
    """Model id (None = the run's model, i.e. the policy under evaluation/training)."""
    client: ClientConfig | None = None
    """Endpoint override (None = the run's client)."""
    sampling: SamplingConfig | None = None
    """Sampling values merged onto the run's sampling."""

    max_turns: int | None = None
    """Max model turns per run (None = no limit)."""
    max_input_tokens: int | None = None
    """Max input tokens per run (None = no limit)."""
    max_output_tokens: int | None = None
    """Max output tokens per run (None = no limit)."""
    max_total_tokens: int | None = None
    """Max total tokens per run (None = no limit)."""
    segments: SegmentsConfig | None = None
    """Run each `interaction()` as a chain of traces under a token budget, joined by
    the model's own summaries (None = one trace per interaction). Limits apply per
    segment."""
    memory: MemoryConfig | None = None
    """A memory the model edits with tags in its replies, kept across segments (None =
    no memory). See `verifiers.v1.memory`."""

    timeout: TimeoutConfig = TimeoutConfig()
    retries: RetryConfig = RetryConfig()

    network_notice: bool = True
    """Include the runtime's network restrictions in the agent's prompt."""

    @field_serializer("sampling")
    def _sampling_overrides(self, sampling: SamplingConfig | None) -> dict | None:
        # An unresolved agent carries only overrides across env-server boundaries.
        # resolve_agent fills every field before the config is recorded on a trace.
        return sampling.model_dump(exclude_unset=True) if sampling is not None else None

    @model_validator(mode="before")
    @classmethod
    def _resolve_harness(cls, data):
        """Narrow a pinned `harness` to its concrete config type by `id` (absent
        stays None = the taskset's default). The lazy import keeps class-body
        `AgentConfig()` defaults constructible while this module initializes."""
        if isinstance(data, dict) and data.get("harness") is not None:
            from verifiers.v1.utils.loaders import (
                harness_config_type,
                narrow_plugin_field,
            )

            narrow_plugin_field(data, "harness", harness_config_type, "bash")
        return data


class WireAgentConfig(AgentConfig):
    """Wire form for trace records: parses without resolving the harness plugin,
    so records round-trip anywhere — the knobs stay readable on the extra-allow
    `WireHarnessConfig` (see `WireTaskData`)."""

    harness: SerializeAsAny[WireHarnessConfig] | None = None

    @model_validator(mode="before")
    @classmethod
    def _resolve_harness(cls, data):
        """Override: a record read resolves no plugins."""
        return data


def agent_config_fields(config: BaseModel) -> dict[str, AgentConfig]:
    """Top-level agent configs, in declaration order, keyed by their field names."""
    return {name: value for name, value in config if isinstance(value, AgentConfig)}


def merge_agent_defaults(config: type[BaseModel], data: Any) -> Any:
    """Merge partial agent overrides onto their declared defaults."""
    if isinstance(data, dict):
        for name, field in config.model_fields.items():
            if isinstance(field.default, AgentConfig) and isinstance(
                data.get(name), dict
            ):
                data[name] = merge_defaults(field.default, data[name])
    return data


def resolve_agent(
    spec: AgentConfig,
    *,
    model: str | None = None,
    client: ClientConfig | None = None,
    sampling: SamplingConfig | None = None,
    harness: HarnessConfig | None = None,
) -> AgentConfig:
    """`spec` with what it leaves unset filled from the run's defaults; its own
    sampling values merge over the run's. The one place a seat's identity resolves,
    for configured agent roles."""
    defaults = sampling or SamplingConfig()
    overrides = (spec.sampling or SamplingConfig()).model_dump(exclude_unset=True)
    merged = {**defaults.model_dump(), **merge_defaults(defaults, overrides)}
    return spec.model_copy(
        update={
            "harness": spec.harness if spec.harness is not None else harness,
            "model": spec.model if spec.model is not None else model,
            "client": spec.client if spec.client is not None else client,
            "sampling": SamplingConfig.model_validate(merged),
        }
    )
