# Environments

An `Env` defines the control flow across one or more `Agents` for a single task. Each completed agent run produces a `Trace`, and the environment collects every trace from the run into an `Episode`.

If your taskset package does not export a custom `Env` and you don't specify `--env.id`, verifiers defaults to `SingleAgentEnv`, which runs a single agent on each task. You can also select one of the built-in environments with `--env.id`:

- `best-of-n` (`BestOfNEnv`): runs `n` independent attempts on the same task, marks which attempt achieved the highest reward (`best`), and records whether any attempt crossed a success threshold (`pass_at_n`), which is useful for rejection sampling and pass@k evaluation.
- `agentic-judge` (`IsolatedAgenticJudgeEnv`): runs a solver agent first, then spins up a judge agent in a fresh runtime to evaluate the result.
- `shared-agentic-judge` (`SharedAgenticJudgeEnv`): runs a solver agent followed by a judge agent that reuses the solver's runtime.
- `user-sim` (`UserSimEnv`): models the user as a second agent and runs a turn-by-turn conversation between the user simulator and the assistant.
- `isolated-verifier` (`IsolatedVerifierEnv`): runs a single solver, transfers only its declared artifacts into a fresh container runtime, and runs deterministic task scoring there without a judge model.

## Custom control flow

To write a custom environment, subclass `vf.Env` and declare an `AgentConfig` field on your `EnvConfig` for each agent role. The field name becomes the CLI and TOML config prefix, such as `--env.solver.model` and `--env.solver.harness.id`:

```python
import verifiers.v1 as vf


class AttemptsConfig(vf.EnvConfig):
    solver: vf.AgentConfig = vf.AgentConfig()
    attempts: int = 2


class AttemptsEnv(vf.Env[AttemptsConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        for _ in range(self.config.attempts):
            await agents.solver.run(task)

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        best = max(trace.reward for trace in episode.traces)
        for trace in episode.traces:
            trace.record_metric("best", float(trace.reward == best))
```

`Env.run(task, agents)` receives the task and pre-initialized agents and orchestrates the episode; it returns `None` because every completed `agent.run(task)` call automatically appends its trace to the episode (with `trace.agent.name` set to the role name). Unless overridden on the role, each agent inherits the evaluation's model and client and the taskset's default harness. Export your `Env` subclass alongside the taskset in your package's `__all__` so verifiers uses it automatically.

When you need episode-level metrics that compare traces across runs, such as marking the best attempt above, implement `finalize(task, episode)`. Because the agents' runtimes have already been released by the time `finalize` runs, any evidence needed for cross-trace scoring should be recorded on the traces during `run`. If an environment includes helper roles that shouldn't be trained (such as a judge or user simulator), mark them in `setup(agents)` with `agents.judge.trainable = False`.

### Role defaults

You can customize a role's default settings directly on its `AgentConfig` field. For example, an answer-only environment that does not need shell tools can default its agent to the `null` harness:

```python
class AnswerConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig(harness={"id": "null"})


class AnswerEnv(vf.Env[AnswerConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        await agents.agent.run(task)
```

CLI flags and TOML settings merge into those defaults, so a user can still switch harnesses at runtime with `--env.agent.harness.id bash`. If a role's default harness depends on a taskset setting, type the config's `taskset` field with that taskset's config class and choose a harness in a Pydantic `after` model validator only when `agent.harness` was left unset, keeping explicit user overrides intact.

### Scripted conversations

When a task has a fixed sequence of follow-up user messages (as opposed to an LLM-driven user simulator, where `user-sim` fits better), you can keep the entire exchange inside a single rollout with `agent.interaction(task)`. Store the opening message in `TaskData.prompt` and the follow-up turns in a field on your `TaskData` subclass:

```python
class ConversationData(vf.TaskData):
    followups: list[str] = []


class ConversationEnv(vf.SingleAgentEnv):
    async def run(self, task: vf.Task[ConversationData], agents: vf.Agents) -> None:
        async with agents.agent.interaction(task) as interaction:
            segment = await interaction.turn()
            for message in task.data.followups:
                if segment.terminated:
                    break
                segment = await interaction.turn(message)
```

This pattern requires a harness that supports conversation resume. Calling `interaction.turn()` with no arguments sends the task's initial `prompt` (or pass a string if `prompt=None`), and each subsequent `turn(message)` runs the next harness segment within the same trace, token budget, and turn budget. Exiting the `async with` block finalizes and scores the rollout even if the script finishes before hitting the agent's turn or token limits. If your task hooks need extra state from the conversation, store it in `interaction.trace.state` or `interaction.trace.info` before leaving the block. See [Agent](agent.md) for more details on driving interactions.

## Isolated deterministic verification

Use `--env.id isolated-verifier` (or export `IsolatedVerifierEnv` from your package) when deterministic scoring needs to run in a clean container that the solver has never touched. It is still a single-agent run: verifiers records one solver trace and does not spin up a second model or harness for grading.

```bash
uv run vf-eval my-task --env.id isolated-verifier --env.agent.runtime.type docker
```

Declare the solver outputs that need to be transferred in `TaskData.artifacts` and write any private test files in `Task.stage_verifier`. Once the solver finishes, the environment collects the declared artifacts, tears down the solver's runtime, provisions a fresh verifier container (using the solver's resolved runtime settings by default, or independent settings under `env.verifier.runtime`), restores the artifacts, and runs your `@vf.reward` and `@vf.metric` methods there. Model-backed task judges are not supported in `isolated-verifier`; see [building environments](building-environments.md#separate-agent-and-grading-sandboxes) for a complete package walkthrough, lifecycle diagram, and validation commands.
