"""Memory: named text entries the model keeps itself across an interaction, segment
switches included, and organizes however it likes.

The model edits it with tags in its reply (after its reasoning; tags inside the
reasoning are ignored):

    <write key="K">text</write>                           append (creates K)
    <rewrite key="K">text</rewrite>                       replace the whole entry
    <replace key="K"><old>a</old><new>b</new></replace>   replace every exact `a`
    <delete key="K">text</delete>                         remove every exact `text`
    <delete key="K"/>                                     remove the entry
    <read key="K"/>                                       show K on the next turn

Every observation the caller sends is prefixed with a [Memory] block: the keys and
their sizes, the results of the last reply's edits, and the entries it asked to read.
The block goes into the observation rather than the system prompt because a turn may
only append to the conversation (a KV-Streams session cannot rewrite its history); old
blocks leave the context with their turns. An entry is capped at `max_key_tokens`
(estimated as characters / 4): an edit that would leave it above the cap and longer
than before is not saved, and the result says to rewrite or delete instead.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING, Any

from verifiers.v1.configs.agent import MemoryConfig
from verifiers.v1.dialects import parse_message
from verifiers.v1.types import AssistantMessage, Messages, UserMessage

if TYPE_CHECKING:
    from verifiers.v1.agent import Interaction, Segment

_OP = re.compile(
    r'<(write|rewrite|replace|delete|read)\s+key\s*=\s*"([^"\n]{1,80})"\s*(?:/>|>(.*?)</\1\s*>)',
    re.DOTALL,
)
_REPLACE = re.compile(r"<old>(.*?)</old>\s*<new>(.*?)</new>", re.DOTALL)


def tokens(text: str) -> int:
    """Estimated tokens: characters / 4, rounded up."""
    return (len(text) + 3) // 4


class Memory:
    """The entries and their edits. The caller owns it (to save and restore it with its
    own state): pass it to `Agent.interaction(..., memory=...)`; `entries` is plain JSON."""

    def __init__(self, entries: dict[str, str] | None = None, max_key_tokens: int = 4096) -> None:
        self.entries: dict[str, str] = dict(entries or {})
        self.max_key_tokens = max_key_tokens
        self.stats: Counter[str] = Counter()  # "<op> <outcome>" -> count
        # The last reply's results and reads, for the next observation: kept here, not on a
        # segment's interaction, so a summary turn's land in the next segment's first one.
        self.results: list[str] = []
        self.reads: list[str] = []

    def index(self) -> str:
        if not self.entries:
            return "[Memory] no entries yet"
        keys = " · ".join(f"{k} ({tokens(v):,} tok)" for k, v in self.entries.items())
        return f"[Memory] {keys}"

    def _full(self, key: str, size: int) -> str:
        return (
            f'"{key}" is full ({size:,} tokens, limit {self.max_key_tokens:,}), so your edit was not '
            f'saved. Free space with <rewrite key="{key}">…</rewrite> (a condensed version) or '
            f'<delete key="{key}">…</delete> (remove text you no longer need), then write again.'
        )

    def _set(self, op: str, key: str, new: str) -> str:
        old = self.entries.get(key, "")
        size = tokens(new)
        if size > self.max_key_tokens and size > tokens(old):
            self.stats[f"{op} full"] += 1
            return self._full(key, size)
        self.entries[key] = new
        self.stats[f"{op} ok"] += 1
        return f'{op} "{key}": saved ({size:,} tok)'

    def apply(self, reply: str) -> tuple[list[str], list[str]]:
        """Apply the reply's tags in order: (one result line per tag, keys to read)."""
        results, reads = [], []
        for m in _OP.finditer(reply or ""):
            op, key, body = m.group(1), m.group(2).strip(), m.group(3)
            old = self.entries.get(key)
            if op == "read":
                if old is None:
                    self.stats["read missing"] += 1
                    results.append(f'read "{key}": no such entry')
                elif key not in reads:
                    self.stats["read ok"] += 1
                    reads.append(key)
            elif op == "write":
                text = (body or "").strip()
                if not text:
                    results.append(f'write "{key}": nothing to write')
                    continue
                results.append(self._set(op, key, f"{old}\n{text}" if old else text))
            elif op == "rewrite":
                results.append(self._set(op, key, (body or "").strip()))
            elif op == "replace":
                pair = _REPLACE.search(body or "")
                if pair is None or old is None:
                    self.stats["replace invalid"] += 1
                    results.append(f'replace "{key}": ' + ("no such entry" if old is None else "needs <old>…</old><new>…</new>"))
                    continue
                needle, text = pair.group(1).strip(), pair.group(2).strip()
                count = old.count(needle) if needle else 0
                if not count:
                    self.stats["replace missing"] += 1
                    results.append(f'replace "{key}": text not found, nothing changed')
                    continue
                line = self._set(op, key, old.replace(needle, text))
                results.append(line + (f", {count} occurrences" if count > 1 and "saved" in line else ""))
            else:  # delete
                if old is None:
                    self.stats["delete missing"] += 1
                    results.append(f'delete "{key}": no such entry')
                elif body is None:
                    del self.entries[key]
                    self.stats["delete entry"] += 1
                    results.append(f'delete "{key}": entry removed')
                else:
                    needle = body.strip()
                    count = old.count(needle) if needle else 0
                    if not count:
                        self.stats["delete missing"] += 1
                        results.append(f'delete "{key}": text not found, nothing changed')
                        continue
                    self.entries[key] = old.replace(needle, "").strip()
                    self.stats["delete ok"] += 1
                    results.append(f'delete "{key}": removed {count} occurrence{"s" if count > 1 else ""} ({tokens(self.entries[key]):,} tok)')
        return results, reads


class MemoryInteraction:
    """One (segment's) interaction with the memory block added to every observation and
    the edits in every reply applied. Segments share one `Memory`, so a summary turn's
    edits count and the next segment opens on them."""

    def __init__(self, inner: Interaction, memory: Memory, opening: str | None = None) -> None:
        self._inner = inner
        self.memory = memory
        self._opening = opening  # the first observation's lead line (a reset or a resumed run)

    @property
    def trace(self):
        return self._inner.trace

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _block(self) -> str:
        lines = [self.memory.index()]
        if self._opening:
            lines.append(self._opening)
            self._opening = None
        lines += self.memory.results
        for key in self.memory.reads:
            if key in self.memory.entries:
                lines.append(f'[Memory "{key}"]\n{self.memory.entries[key]}\n[/Memory "{key}"]')
        self.memory.results, self.memory.reads = [], []
        return "\n".join(lines)

    def _with_block(self, message: str | Messages) -> str | Messages:
        block = self._block()
        if isinstance(message, str):
            return f"{block}\n\n{message}"
        messages = [parse_message(m) if isinstance(m, dict) else m for m in message]
        last = messages[-1] if messages else None
        if isinstance(last, UserMessage) and isinstance(last.content, str):
            return [*messages[:-1], last.model_copy(update={"content": f"{block}\n\n{last.content}"})]
        return [*messages, UserMessage(content=block)]

    async def turn(self, message: str | Messages | None = None, *, max_tokens: int | None = None) -> Segment:
        if message is not None:
            message = self._with_block(message)
        segment = await self._inner.turn(message, max_tokens=max_tokens)
        if segment.terminated:
            return segment
        self.memory.results, self.memory.reads = self.memory.apply(segment.last_reply)
        if not self.memory.results and not self.memory.reads:
            reasoning = "".join(m.reasoning_content or "" for m in segment.messages if isinstance(m, AssistantMessage))
            if _OP.search(reasoning):
                self.memory.stats["ignored in reasoning"] += 1
                self.memory.results = ["memory tags inside your reasoning were ignored: put them in your reply"]
        return segment


def memory_system_prompt(config: MemoryConfig, base: str | None) -> str:
    note = config.prompt.replace("{max_key_tokens}", f"{config.max_key_tokens:,}")
    return f"{base or ''}\n\n{note}".lstrip()


RESET = "Your context was reset and your memory was kept: read the entries you need before you continue."
