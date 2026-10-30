"""The built-in `trace` judge: what it shows the model, how the verdict becomes a reward,
config-typed subclasses, and `on_error`."""

import pytest

import verifiers.v1 as vf
from verifiers.v1.judge import Judge, JudgeResponse, judge_config_cls


@pytest.fixture
def judge_reply(monkeypatch):
    """Fake the model call with a fixed reply text; record each prompt."""
    calls: dict = {"prompts": [], "text": '{"reason": "it looped", "score": 7}'}

    async def fake_complete(self, messages, *, trace=None, schema=None, parse=None, **sampling):
        calls["prompts"].append(messages)
        response = JudgeResponse(text=calls["text"])
        if parse is not None:
            response.parsed = parse(response)
        return response

    monkeypatch.setattr(Judge, "complete", fake_complete)
    return calls


async def test_scores_the_rendered_trace(judge_reply, make_full_trace):
    trace = make_full_trace()
    judge = vf.TraceJudge()
    assert await judge.score(trace.task.data, trace) == 0.7
    prompt = judge_reply["prompts"][0]
    assert "SECRET REASONING" in prompt and "TOOL RESULT" in prompt and "[tool_call search(" in prompt
    assert "Capital of France?" in prompt and '"score": <integer 0 to 10>' in prompt
    no_reasoning = vf.TraceJudge(vf.TraceJudgeConfig(reasoning=False))
    await no_reasoning.score(trace.task.data, trace)
    assert "SECRET REASONING" not in judge_reply["prompts"][1]


async def test_plugged_judge_adds_a_weighted_reward(judge_reply, make_full_trace):
    config = vf.TaskConfig.model_validate({"judges": [{"id": "trace", "weight": 0.5, "criteria": "LOOPS?"}]})
    assert isinstance(config.judges[0], vf.TraceJudgeConfig)
    trace = make_full_trace()
    task = vf.Task(trace.task.data, config)
    await task.score(trace)
    assert trace.rewards["trace"].score == 0.7 and trace.reward == pytest.approx(0.35)
    assert "LOOPS?" in judge_reply["prompts"][0]


async def test_the_trace_is_cut_to_its_end(judge_reply, make_full_trace):
    trace = make_full_trace()
    await vf.TraceJudge(vf.TraceJudgeConfig(max_chars=20)).score(trace.task.data, trace)
    assert "earlier characters cut" in judge_reply["prompts"][0] and "It is Paris." in judge_reply["prompts"][0]


@pytest.mark.parametrize("text", ["no json here", '{"reason": "x", "score": 11}'])
async def test_a_bad_verdict_raises_or_is_skipped(judge_reply, make_full_trace, text):
    judge_reply["text"] = text
    trace = make_full_trace()
    with pytest.raises(ValueError):
        await vf.TraceJudge().score(trace.task.data, trace)
    skip = vf.TraceJudge(vf.TraceJudgeConfig(on_error="skip"))
    assert await skip.score(trace.task.data, trace) == {}
    assert trace.metrics["trace/error"] == 1.0


def test_subclasses_resolve_their_config():
    class MyConfig(vf.TraceJudgeConfig):
        id: vf.ID = "mine"

    class Typed(vf.TraceJudge[MyConfig]):
        pass

    class Plain(vf.TraceJudge):
        pass

    assert judge_config_cls(vf.TraceJudge) is vf.TraceJudgeConfig
    assert judge_config_cls(Plain) is vf.TraceJudgeConfig
    assert judge_config_cls(Typed) is MyConfig and isinstance(Typed().config, MyConfig)
    assert judge_config_cls(vf.RubricJudge) is vf.RubricJudgeConfig


def test_a_dotted_id_names_a_judge_module_in_a_package(tmp_path, monkeypatch):
    package = tmp_path / "my_env"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "judge.py").write_text(
        "import verifiers.v1 as vf\n\n"
        "class MyJudgeConfig(vf.TraceJudgeConfig):\n    id: vf.ID = 'my_env.judge'\n\n"
        "class MyJudge(vf.TraceJudge[MyJudgeConfig]):\n    pass\n\n"
        "__all__ = ['MyJudge']\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config = vf.TaskConfig.model_validate({"judges": [{"id": "my_env.judge", "scale": 4}]})
    assert type(config.judges[0]).__name__ == "MyJudgeConfig" and config.judges[0].scale == 4
