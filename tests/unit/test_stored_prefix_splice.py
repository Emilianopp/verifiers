"""A bridge-less turn keeps the engine's stored prefix tokens instead of a re-render.

A full render re-tokenizes history from messages; when that differs from what was sampled
(another BPE split, a reply without the template's newline before ``</think>``) a streaming
session sees a forked prompt and reopens, resurrecting evicted turns.
"""

import pytest
from renderers import RenderedTokens

from verifiers.v1.clients.train import splice_stored_prefix

STOP = 9


def _rendered(ids, indices, roles):
    return RenderedTokens(
        token_ids=ids,
        message_indices=indices,
        sampled_mask=[False] * len(ids),
        is_content=[True] * len(ids),
        message_roles=roles,
        message_tool_names=[None] * len(roles),
    )


# Canonical: user [1, 2, STOP, 0], assistant [3, 4, STOP, 0], new user [5, STOP, 0], gen prompt [7].
FULL = _rendered(
    [1, 2, STOP, 0, 3, 4, STOP, 0, 5, STOP, 0, 7],
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, -1],
    ["user", "assistant", "user"],
)

def test_keeps_sampled_tokens_and_appends_only_the_tail():
    stored = [1, 2, STOP, 0, 3, 44, 4, STOP]  # the reply sampled as another split; ends at its stop
    spliced = splice_stored_prefix(stored, FULL, 2, {STOP})
    assert spliced.token_ids == stored + [0, 5, STOP, 0, 7]
    assert spliced.message_indices == [-1] * 8 + [-1, 0, 0, 0, -1]
    assert spliced.message_roles == ["user"]
    assert len(spliced.sampled_mask) == len(spliced.is_content) == len(spliced.token_ids)


def test_stored_scaffolding_after_the_stop_must_match():
    assert splice_stored_prefix([1, 2, STOP, 0, 3, STOP, 0], FULL, 2, {STOP}).token_ids == [
        1, 2, STOP, 0, 3, STOP, 0, 5, STOP, 0, 7,
    ]
    assert splice_stored_prefix([1, 2, STOP, 0, 3, STOP, 8], FULL, 2, {STOP}) is None


def test_history_the_render_rewrites_is_kept_as_stored():
    """A template that drops old thinking renders the history shorter; the tail is still
    found by attribution, and the stored history is what the engine holds."""
    full = _rendered([1, 2, STOP, 0, 4, STOP, 0, 5, STOP, 0, 7], [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, -1], ["user", "assistant", "user"])
    stored = [1, 2, STOP, 0, 3, 3, 4, STOP]
    assert splice_stored_prefix(stored, full, 2, {STOP}).token_ids == stored + [0, 5, STOP, 0, 7]


@pytest.mark.parametrize(
    "stored",
    [
        [1, 2, STOP, 0, 3, 4],  # reply cut off without a stop: anchored on the user's close
        [],
        [1, 2],
    ],
)
def test_refuses_what_it_cannot_align(stored):
    assert splice_stored_prefix(stored, FULL, 2, {STOP}) is None


def test_qwen35_reply_without_newline_before_think_end():
    """The second reopen seen in a Craftax run: the model closed ``</think>`` straight after
    its reasoning and stopped with no answer; the template renders ``\\n</think>\\n\\n``."""
    from renderers import create_renderer
    from renderers.base import load_tokenizer
    from renderers.configs import config_from_name

    tok = load_tokenizer("Qwen/Qwen3.5-4B")
    renderer = create_renderer(
        tok,
        config_from_name("qwen3.5").model_copy(
            update={"enable_thinking": True, "thinking_retention": "all"}
        ),
    )
    messages = [
        {"role": "system", "content": "Play."},
        {"role": "user", "content": "obs 1"},
    ]
    prompt = renderer.render_ids(messages, add_generation_prompt=True)
    completion = list(tok.encode("Wood at 0,26</think>", add_special_tokens=False)) + [
        renderer._im_end
    ]
    parsed = renderer.parse_response(completion, prompt_ids=prompt)
    messages += [
        {"role": "assistant", "content": parsed.content, "reasoning_content": parsed.reasoning_content},
        {"role": "user", "content": "obs 2"},
    ]
    full = renderer.render(messages, add_generation_prompt=True)
    assert full.token_ids[: len(prompt) + len(completion)] != prompt + completion  # the fork
    spliced = splice_stored_prefix(
        prompt + completion, full, 3, set(renderer.get_stop_token_ids())
    )
    assert spliced is not None
    assert spliced.token_ids[: len(prompt) + len(completion)] == prompt + completion
    assert tok.decode(spliced.token_ids[len(prompt) + len(completion) :]) == (
        "\n<|im_start|>user\nobs 2<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )
