"""Memory: the model's tag edits, the [Memory] block on every observation, and a store
shared across segments (a summary turn's edits count, the next segment opens on them)."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1.agent import Segment
from verifiers.v1.configs.agent import MemoryConfig, SegmentsConfig
from verifiers.v1.memory import RESET, Memory, MemoryInteraction, memory_system_prompt
from verifiers.v1.segments import segmented


def test_write_appends_and_creates():
    m = Memory()
    results, reads = m.apply('<write key="map">tree at (3,4)</write> then <write key="map">water at (5,5)</write>')
    assert m.entries == {"map": "tree at (3,4)\nwater at (5,5)"}
    assert results == ['write "map": saved (4 tok)', 'write "map": saved (7 tok)'] and reads == []


def test_replace_delete_rewrite_and_read():
    m = Memory({"rules": "saplings grow trees\nstone needs a pickaxe\nsaplings grow trees"})
    results, reads = m.apply(
        '<replace key="rules"><old>saplings grow trees</old><new>saplings do not give wood</new></replace>'
        '<delete key="rules">stone needs a pickaxe</delete><read key="rules"/><read key="nope"/>'
    )
    assert m.entries["rules"] == "saplings do not give wood\n\nsaplings do not give wood"
    assert results[0].endswith("2 occurrences") and results[1].startswith('delete "rules": removed 1 occurrence')
    assert results[2] == 'read "nope": no such entry' and reads == ["rules"]
    m.apply('<rewrite key="rules">only this</rewrite><delete key="gone"/>')
    assert m.entries["rules"] == "only this"
    m.apply('<delete key="rules"/>')
    assert m.entries == {}


def test_missing_text_changes_nothing():
    m = Memory({"a": "x"})
    results, _ = m.apply('<replace key="a"><old>y</old><new>z</new></replace><delete key="a">y</delete>')
    assert m.entries == {"a": "x"}
    assert results == ['replace "a": text not found, nothing changed', 'delete "a": text not found, nothing changed']


def test_a_full_entry_refuses_growth_but_takes_a_rewrite():
    m = Memory({"log": "x" * 40}, max_key_tokens=10)
    results, _ = m.apply('<write key="log">more</write>')
    assert m.entries["log"] == "x" * 40
    assert '"log" is full' in results[0] and "<rewrite" in results[0] and "<delete" in results[0]
    results, _ = m.apply('<rewrite key="log">short</rewrite>')
    assert m.entries["log"] == "short" and results[0].startswith('rewrite "log": saved')


def test_the_prompt_states_the_limit():
    assert "4,096 tokens" in memory_system_prompt(MemoryConfig(), "sys")


EMPTY = Memory().index()


def _task():
    return vf.Task(vf.TaskData(idx=0, prompt=None, system_prompt="sys"))


class FakeTurns:
    """Every reply is `replies.pop(0)` ("content|think:reasoning"); records what each turn was sent."""

    def __init__(self, replies: list[str], tokens: int = 0):
        self.replies, self.tokens = replies, tokens
        self.sent: list[str] = []
        self.systems: list[str] = []

    @asynccontextmanager
    async def open(self, task):
        self.systems.append(task.data.system_prompt)
        trace = SimpleNamespace(nodes=[], calls=[], num_total_tokens=0, info={})
        fake = self

        class Inner:
            async def turn(self, message, max_tokens=None):
                fake.sent.append(message)
                trace.num_total_tokens += fake.tokens
                content, _, reasoning = fake.replies.pop(0).partition("|think:")
                return Segment(messages=[vf.AssistantMessage(content=content or None, reasoning_content=reasoning or None)])

        inner = Inner()
        inner.trace = trace
        yield inner


@pytest.mark.asyncio
async def test_every_observation_carries_the_block_and_last_results():
    fake = FakeTurns(['<write key="plan">get wood</write><read key="plan"/><action>DO</action>', "<action>DO</action>"])
    store = Memory()
    async with fake.open(_task()) as inner:
        interaction = MemoryInteraction(inner, store, note="NOTE")
        await interaction.turn("obs 1")
        await interaction.turn("obs 2")
    assert fake.sent[0] == f"{EMPTY}\n\nobs 1\n\nNOTE"
    assert fake.sent[1].endswith("obs 2\n\nNOTE")
    assert fake.sent[1].startswith('[Memory] plan (2 tok)\nwrite "plan": saved (2 tok)\n[Memory "plan"]\nget wood\n[/Memory "plan"]\n\nobs 2')


@pytest.mark.asyncio
async def test_tags_in_reasoning_are_ignored_and_said_so():
    fake = FakeTurns(['<action>DO</action>|think:<write key="a">x</write>', "<action>DO</action>"])
    async with fake.open(_task()) as inner:
        interaction = MemoryInteraction(inner, store := Memory())
        await interaction.turn("obs 1")
        await interaction.turn("obs 2")
    assert store.entries == {} and "ignored" in fake.sent[1]


@pytest.mark.asyncio
async def test_segments_share_the_store_and_the_summary_turn_edits_it():
    config = SegmentsConfig(new_tokens=100, summary_tokens=10, summary_header="HEADER")
    fake = FakeTurns(
        ["<action>DO</action>", '<write key="map">table at (1,2)</write><summary>s</summary>', "<action>DO</action>"],
        tokens=95,
    )
    store = Memory()

    @asynccontextmanager
    async def open_segment(task):  # as Agent.interaction wraps each segment
        async with fake.open(task) as inner:
            yield MemoryInteraction(inner, store, RESET if store.entries else None)

    async with segmented(config, _task(), open_segment) as interaction:
        await interaction.turn("obs 1")
        await interaction.turn("obs 2")  # switch: the summary turn writes, the next segment opens
    assert store.entries == {"map": "table at (1,2)"}
    assert fake.sent[1].startswith(f"{EMPTY}\n\nobs 2\n\n")  # the summary request
    assert fake.sent[2].startswith(f'[Memory] map (4 tok)\n{RESET}\nwrite "map": saved (4 tok)\n\nobs 2')
