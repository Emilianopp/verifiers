# Runtime

A `Runtime` provides command and file access inside a sandbox with immutable user, `HOME`, and environment bindings. Task and harness hooks receive it as `runtime: vf.Runtime`, while configuration fields such as `agent.runtime` describe the runtime policy used to provision the sandbox (see [Runtimes](runtimes.md) for backend capabilities, resources, and network rules).

Calling `runtime.with_user(...)` or `runtime.with_env(...)` returns a new view over the same underlying resource without mutating existing views. Trusted task setup and grading use the default-user view, while agent execution and colocated tools use the task's `agent_user` view. Note that a view is an execution handle rather than a security boundary: user isolation is enforced by OS permissions inside the container backend.

Under the hood, a `RuntimeBackend` (such as `DockerBackend` or `PrimeBackend`) owns the provider lifecycle, including container startup, network policy, and teardown. Calling `backend.bind()` creates a `Runtime` view without provisioning a new resource, and `runtime.backend` exposes the backend when explicit infrastructure operations are needed. `Agent.provision()` lends a runtime whose cleanup remains owned by the caller's context.

## Deployment and services

A `Deployment` coordinates all execution resources for a single rollout attempt: provisioning, placement, connectivity, and cleanup of the harness environment, the task workspace, supporting sidecar services, and fresh verification targets.

The **workspace** (`deployment.workspace`) is the `Runtime` where task commands run and task files live; setting `agent.deployment.mode = "colocated"` places the harness and colocated tools inside that same workspace.

When a task defines **Compose services** by returning a `ComposeSpec` from `deployment_spec()`, the deployment brings up the multi-container stack and exposes the designated main service as `deployment.workspace` (for example, running `app` as the workspace while a `postgres` sidecar runs alongside it). Without a `ComposeSpec`, the deployment manages a single runtime.

Task hooks can request `deployment` by parameter name and call `deployment.target(name)` to get a `Runtime` view for a specific named service, and `finalize_services()` collects sidecar artifacts before isolated verification. See [Harbor's Docker Compose support](harbor.md#docker-compose) for provider requirements; on local Docker, Compose tasks require `agent.deployment.trust_compose = true` because task-authored Compose files can mount host paths and request Docker privileges.

To validate an isolated task without calling a model, finalize the submission, collect its artifacts, and call `vf.grade_submission(task, trace, runtime)`. This runs the task's normal grader, verifier image, timeouts, and retries in a fresh verification target without taking ownership of the supplied solver runtime.
