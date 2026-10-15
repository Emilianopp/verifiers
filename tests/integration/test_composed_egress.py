"""A composed deployment on a Prime VM restricts only the workspace container."""

import json
import os

import pytest

import verifiers.v1 as vf
from verifiers.v1.configs.agent import TimeoutConfig

pytestmark = pytest.mark.skipif(
    not os.environ.get("PRIME_API_KEY"), reason="needs PRIME_API_KEY"
)

COMPOSE = """
services:
  sidecar:
    image: python:3.11-slim
    command: [python, -m, http.server, "8000"]
    healthcheck:
      test: [CMD, python, -c, "import urllib.request; urllib.request.urlopen('http://localhost:8000')"]
      interval: 1s
      timeout: 2s
      retries: 60
"""

FETCH = """
import sys, urllib.request
try:
    print(urllib.request.urlopen(sys.argv[1], timeout=15).status)
except Exception as error:
    print(type(error).__name__)
"""

PROBE = """
import json, socket, urllib.request

def fetch(url):
    try:
        return urllib.request.urlopen(url, timeout=15).status
    except Exception as error:
        return type(error).__name__

def connect(host, port):
    try:
        socket.create_connection((host, port), timeout=8).close()
        return "connected"
    except Exception as error:
        return type(error).__name__

def resolve(name):
    try:
        return socket.gethostbyname(name)
    except Exception as error:
        return type(error).__name__

json.dump(
    {
        "sidecar": fetch("http://sidecar:8000"),
        "public": fetch("https://example.com"),
        "socket": connect("1.1.1.1", 443),
        "dns": resolve("example.com"),
    },
    open("/tmp/probe.json", "w"),
)
"""


class ProbeHarness(vf.Harness):
    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data):
        return await runtime.run_program(["python", "-c", PROBE], {})


class ComposedData(vf.TaskData):
    compose_dir: str


class ComposedTask(vf.Task[ComposedData, vf.State, vf.TaskConfig]):
    def deployment_spec(self) -> vf.ComposeSpec:
        return vf.ComposeSpec(directory=self.data.compose_dir)

    async def finalize(self, trace, runtime, deployment):
        trace.info["workspace"] = json.loads(await runtime.read("/tmp/probe.json"))
        sidecar = await deployment.target("sidecar").run(
            ["python", "-c", FETCH, "https://example.com"], {}
        )
        trace.info["sidecar"] = sidecar.stdout.strip()


async def test_composed_policy_restricts_only_the_workspace(tmp_path):
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment" / "docker-compose.yaml").write_text(COMPOSE)
    task = ComposedTask(
        ComposedData(
            prompt="Probe the network",
            compose_dir=str(tmp_path),
            network_allow=[],
        )
    )
    config = vf.AgentConfig(
        model="unused",
        runtime=vf.PrimeConfig(cpu=2, memory=4, disk=10, labels=["vf-integration"]),
        timeout=TimeoutConfig(setup=900, finalize=120, scoring=60),
    )
    async with vf.Agent(config) as agent:
        agent.harness = ProbeHarness(vf.HarnessConfig())
        trace = await agent.run(task)
    assert trace.ok, trace.errors
    workspace = trace.info["workspace"]
    assert workspace["sidecar"] == 200
    assert workspace["public"] != 200
    assert workspace["socket"] != "connected"
    assert workspace["dns"] == "gaierror"
    assert trace.info["sidecar"] == "200"
