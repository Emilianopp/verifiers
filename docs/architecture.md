# Architecture

verifiers is built out of the following parts:

A server-backed evaluation or prime-rl **orchestrator** creates worker processes and distributes rollout requests among them. The client owns the taskset: it loads the tasks once and ships each task to the workers, which own the runtime containing the agent(s), score the result, and return the episode and its traces.

The orchestrator and workers are managed by verifiers and prime-rl themselves and thus offer few configurable knobs.

The **rollout** is the executable combination of one loaded task, the harness, and any tools. Each rollout has an independent trace and runtime state. verifiers supports several [runtimes](runtimes.md) that you can use for most tasksets:

- The `subprocess` runtime runs rollouts in local Python subprocesses. It is meant for debugging, since processes share your machine and can have side effects on one another (such as altering a harness's config files).
- The `docker` runtime runs rollouts in Docker containers on your local machine.
- The `podman` runtime uses the same container implementation through the Podman CLI for hosts without Docker.
- The `apptainer` runtime runs rollouts in unprivileged Apptainer instances on the host network, as on HPC clusters, without network egress filtering.
- Sandbox runtimes, such as `prime`, `modal`, or `e2b`, run remotely and are meant for production workloads like training or high-concurrency evaluation.

For offline Docker/Podman use on Linux, cache the task image first and, if it lacks Python 3, `docker.io/library/python:3.11-alpine` for host callbacks. Network-restricted execution also needs the cached `localhost/verifiers-network:1` image, which is built during the first online startup.

The harness runs inside the rollout runtime to interact with the task. Rather than calling the model provider directly, the harness sends model traffic through verifiers' **interception server** over a local connection or [Prime Tunnel](https://docs.primeintellect.ai/sandboxes/tunnel).

The interception server receives these requests and forwards them to the upstream provider using the API surface that the harness expects, such as OpenAI Responses for Codex or Anthropic Messages for Claude Code.

Routing traffic through the interception server enables several things beyond forwarding the provider response:

- Building traces live so trajectories and token usage are recorded as they happen
- Applying sampling parameters in harnesses that don't expose those settings directly
- Running the task's [stop and interception hooks](tasksets.md#stops-and-interception) to inspect or rewrite tool responses and server-side web search results (for example, to block reward hacks)
