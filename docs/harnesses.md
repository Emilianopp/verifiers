# Harnesses

Use a built-in harness when possible. Options include Claude Code, Codex,
`bash` for shell tools, `browser-use` for browser tasks, and `null` for basic
chat. `null` has no built-in shell but can use task MCP tools.

Use `uv run vf-eval my-task --env.agent.harness.id codex --help` to inspect a
harness's settings. Most third-party harnesses require a container. Check support
for each feature you need: MCP tools, follow-up messages, image input, and skills
are separate features.

## Bash harness

`bash` supports shell commands, editing, MCP tools, conversation resume, and
tool interception. It can run on `subprocess` for trusted local work, or in a
container. It does not support harness skills.

Context compaction is enabled by default. When the provider reports the context
window, automatic compaction starts when about 16,000 tokens remain.
Set a fixed threshold when needed:

```toml
[env.agent.harness]
id = "bash"
compaction = { summarize_at_tokens = 100000 }
```

Set `compaction = "None"` to disable it. Search is off by default; enabling
`search = true` requires `SERPER_API_KEY` on the evaluator.

## A minimal harness implementation

```python
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace


class MyHarnessConfig(HarnessConfig):
    # Users can override these settings in CLI flags or TOML.
    version: str = "0.0.1"


class MyHarness(Harness[MyHarnessConfig]):
    # Pass the task's system prompt as a system message.
    APPENDS_SYSTEM_PROMPT = True

    async def setup(self, runtime: Runtime) -> None:
        # Replace this with the program's installation command.
        await runtime.run(["sh", "-c", "echo installing..."], {})

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ProgramResult:
        system, prompt = self.resolve_text_prompt(data)

        # Route model calls through verifiers so they are recorded.
        env = {
            **self.config.resolved_env,
            "HARNESS_BASE_URL": endpoint,
            "HARNESS_API_KEY": secret,
            "HARNESS_BASE_MODEL": ctx.model,
            "HARNESS_SYSTEM_PROMPT": system or "",
        }
        return await runtime.run_program(["<HARNESS_BINARY>", str(prompt or "")], env)
```

Replace the placeholder binary, installation command, and environment variable
names with those your program uses. Every model call must use the supplied
`endpoint` and `secret` so verifiers can record it. Include
`self.config.resolved_env` to support credentials passed through `forward_env`.
When the task sets `agent_user`, harness setup uses that user too. Install into
writable paths and leave system package installation to the image or task setup.

Capability flags describe behavior you must implement:

- `SUPPORTS_MCP`: pass `mcp_urls` to the program.
- `SUPPORTS_RESUME`: restart the program from the conversation so far. Programs with saved sessions should implement `resume` or a session instead.
- `SUPPORTS_SKILLS`: use `install_skills` to place skills where the program looks for them.
- `SUPPORTS_TOOL_INTERCEPTION`: check tool calls through the rollout's tool gate before executing them.
- `APPENDS_SYSTEM_PROMPT`: pass the system prompt as a system message. Otherwise, it is added to the first user message.

For an ACP program, reuse `vf.ACPHarness` in
[`acp/__init__.py`](../../verifiers/v1/acp/__init__.py).

Live steering requires a harness session that implements `steer`. Conversation
resume support alone does not provide it; see [message delivery](agent.md#messages-during-an-active-turn).

Return the main program's `ProgramResult`; verifiers builds the trace. If the
harness creates files or sessions outside the temporary runtime, remove them in
`cleanup(trace, runtime)`. Cleanup runs after scoring, before the runtime is
released, and must be safe to call more than once. It also runs when the
environment owns the runtime.
