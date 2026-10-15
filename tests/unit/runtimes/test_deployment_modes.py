import asyncio
import base64
import io
import json
import subprocess
import sys
import tarfile

import pytest

from verifiers.v1.mcp.toolset import ToolsetConfig
from verifiers.v1.runtimes import SubprocessConfig
from verifiers.v1.runtimes.compose import ComposeSpec
from verifiers.v1.runtimes.deployment import resolve_mode
from verifiers.v1.runtimes.docker.host_egress import bundle
from verifiers.v1.runtimes.strategies import _deployment_strategy

SPEC = ComposeSpec(directory="/task")


def test_unset_mode_follows_the_task():
    assert resolve_mode(None, None) == "colocated"
    assert resolve_mode(None, SPEC) == "composed"


def test_explicit_mode_must_fit_the_task():
    with pytest.raises(ValueError, match="Compose services"):
        resolve_mode("colocated", SPEC)
    with pytest.raises(ValueError, match="needs a task with Compose"):
        resolve_mode("composed", None)
    with pytest.raises(ValueError, match="Docker, Prime, or Modal"):
        _deployment_strategy("composed", SubprocessConfig(), SPEC)


def test_service_placement_excludes_other_placements():
    assert ToolsetConfig(service="tools").service == "tools"
    with pytest.raises(ValueError):
        ToolsetConfig(service="tools", colocated=True)
    with pytest.raises(ValueError):
        ToolsetConfig(service="tools", url="http://example.com/mcp")


def test_host_proxy_rereads_its_policy_and_requires_credentials(tmp_path):
    with tarfile.open(fileobj=io.BytesIO(bundle())) as archive:
        archive.extractall(tmp_path / "pkg", filter="data")
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {"allow": [], "block": ["*"], "routes": [], "allow_non_global": False}
        )
    )
    info = tmp_path / "info.json"
    process = subprocess.Popen(
        [
            sys.executable,
            str(tmp_path / "pkg/vf_egress/serve.py"),
            "--lib",
            str(tmp_path / "lib"),
            "--host",
            "127.0.0.1",
            "--policy",
            str(policy),
            "--info",
            str(info),
        ]
    )

    async def connect(target: str, token: str | None) -> bytes:
        port = json.loads(info.read_text())["port"]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        header = ""
        if token is not None:
            credentials = base64.b64encode(f"verifiers:{token}".encode()).decode()
            header = f"Proxy-Authorization: Basic {credentials}\r\n"
        writer.write(
            f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n{header}\r\n".encode()
        )
        await writer.drain()
        line = await reader.readline()
        writer.close()
        return line

    async def check() -> None:
        for _ in range(100):
            if info.exists():
                break
            await asyncio.sleep(0.1)
        token = json.loads(info.read_text())["token"]
        assert b" 407 " in await connect("example.com:443", None)
        assert b" 403 " in await connect("example.com:443", token)
        # A replaced policy file applies to the next connection; loopback stays refused.
        replacement = tmp_path / "policy.next"
        replacement.write_text(
            json.dumps(
                {"allow": ["*"], "block": [], "routes": [], "allow_non_global": True}
            )
        )
        replacement.replace(policy)
        assert b" 403 " in await connect("127.0.0.1:443", token)

    try:
        asyncio.run(check())
    finally:
        process.terminate()
        process.wait(timeout=10)
