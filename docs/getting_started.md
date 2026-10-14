# Installation

verifiers runs locally with `uv`. Install it, clone the repo, and sync dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/PrimeIntellect-ai/verifiers.git
cd verifiers
uv sync
```

For model evaluations, use [Prime-RL's `uv run eval`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/eval.md)
from a Prime-RL installation. The bundled `vf-eval` command is still available,
but tells users it will be deprecated and directs them there.

The Verifiers checkout installs its example tasksets. To check an example's
configuration with the bundled CLI:

```bash
uv run vf-eval reverse-text --env.agent.harness.id null \
  --env.agent.runtime.type subprocess --no-push --dry-run
```

This dry run does not call a model. The [bundled evaluation reference](evaluation.md)
describes `vf-eval`'s settings and output files. Prime-RL has its own run config,
resume behavior, and output layout; use its guide for those runs.

Create new packages with `uv run vf-init <name>` and install them with
`uv pip install -e environments/<package_dir>`. Other tasksets must also be
installed before `vf-eval` can load their IDs. Harbor needs Python 3.12+ and the
`harbor` extra (`uv sync --extra harbor` in this checkout).

## Commands

Verifiers installs these six commands. Run them with `uv run` in the environment
where Verifiers and the taskset package are installed.

| Command | Use |
| --- | --- |
| [`vf-init <name>`](building-environments.md#create-a-package) | Scaffold an environment package; add `-T` for tools or `-H` for a harness |
| [`vf-validate <taskset>`](debugging.md#before-running-an-agent) | Check setup, a known solution, or the untouched task |
| [`vf-debug <taskset>`](debugging.md#before-running-an-agent) | Run a command or script inside the prepared task runtime |
| [`vf-replay <run-dir>`](debugging.md#re-score-without-rerunning-the-agent) | Re-score saved `vf-eval` traces; cannot recreate their sandbox |
| [`vf-gepa <taskset>`](gepa.md) | Optimize a system prompt against task rewards |
| [`vf-eval <taskset>`](evaluation.md) | Run the bundled evaluator; planned for deprecation in favor of Prime-RL |

Use `<command> --help` for its flags. Include the taskset and harness IDs when
reading evaluation help so their options appear. Validation and debugging use
`--taskset.*` and `--runtime.*`; evaluation uses `--env.taskset.*` and
`--env.<agent>.runtime.*`.

## Skills

The repository's [`skills/`](../../skills) and [`AGENTS.md`](../../AGENTS.md) give coding agents instructions for building, evaluating, and debugging environments. These docs explain the same features for people.
