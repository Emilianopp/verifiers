"""Execution policy for one container on a provider VM that hosts Docker.

A provider's network policy covers its whole VM, so it cannot leave Compose services
open while restricting the workspace. Instead the VM host runs verifiers' egress proxy
and the workspace's network namespace is cut down to that proxy, its Compose peers and
loopback. Other services and the VM keep the VM's network.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
import io
import json
import shlex
import tarfile
import uuid
from ipaddress import ip_interface
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from verifiers.v1.configs import runtime as runtime_config
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.docker import egress

if TYPE_CHECKING:
    from verifiers.v1.runtimes.base import RuntimeBackend

ROOT = "/opt/verifiers/egress"
# The proxy's own dependencies, at the evaluator's versions.
REQUIREMENTS = ("h11", "pydantic", "prime-pydantic-config")
HOST_TOOLS = ("python3", "iptables", "ip6tables", "nsenter")

SERVE = '''"""Serve verifiers' egress proxy for one restricted container on this host."""

import argparse
import asyncio
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--lib", required=True)
parser.add_argument("--host", required=True)
parser.add_argument("--policy", required=True)
parser.add_argument("--info", required=True)
args = parser.parse_args()
sys.path[:0] = [args.lib, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]

from vf_egress.egress import EgressProxy, NetworkPolicy  # noqa: E402
from vf_egress.policy import NetworkPolicyConfig  # noqa: E402


class Policies:
    """The policy file's current contents; replaced files are reread."""

    def __init__(self, path):
        self.path = path
        self.stamp = None
        self.policy = None

    def current(self):
        stat = os.stat(self.path)
        stamp = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        if stamp != self.stamp:
            with open(self.path) as file:
                data = json.load(file)
            self.policy = NetworkPolicy(
                NetworkPolicyConfig(allow=data["allow"], block=data["block"]),
                data["routes"],
                allow_non_global=data["allow_non_global"],
            )
            self.stamp = stamp
        return self.policy


class HostProxy(EgressProxy):
    def __init__(self, policies):
        super().__init__(policies.current())
        self.policies = policies

    async def _runtime(self, reader, writer):
        # Each connection uses the latest policy; an unreadable policy fails closed.
        try:
            self.policy = self.policies.current()
        except Exception:
            writer.close()
            return
        await super()._runtime(reader, writer)


async def main():
    proxy = HostProxy(Policies(args.policy))
    await proxy.start(args.host)
    temporary = f"{args.info}.tmp"
    with open(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as file:
        json.dump({"port": proxy.port, "token": proxy.token, "pid": os.getpid()}, file)
    os.replace(temporary, args.info)
    await asyncio.Event().wait()


asyncio.run(main())
'''

# Run on the VM host as root. Arguments: container PID, proxy address and port,
# then the gateways and subnets of the container's networks.
CUT = r"""set -eu
PID=$1; PROXY=$2; PORT=$3; GATEWAYS=$4; SUBNETS=$5
n() { nsenter -t "$PID" -n "$@"; }
n iptables -w -F OUTPUT
# Docker's embedded DNS forwards unknown names outside the namespace.
n iptables -w -A OUTPUT -d 127.0.0.11 -j REJECT
n iptables -w -A OUTPUT -o lo -j ACCEPT
n iptables -w -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED --ctdir REPLY -j ACCEPT
n iptables -w -A OUTPUT -d "$PROXY" -p tcp --dport "$PORT" -j ACCEPT
for gateway in $GATEWAYS; do n iptables -w -A OUTPUT -d "$gateway" -j REJECT; done
for subnet in $SUBNETS; do n iptables -w -A OUTPUT -d "$subnet" -j ACCEPT; done
n iptables -w -A OUTPUT -j REJECT
n iptables -w -F INPUT
n iptables -w -A INPUT -i lo -j ACCEPT
n iptables -w -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
for gateway in $GATEWAYS; do n iptables -w -A INPUT -s "$gateway" -j REJECT; done
for subnet in $SUBNETS; do n iptables -w -A INPUT -s "$subnet" -j ACCEPT; done
n iptables -w -A INPUT -j REJECT
for chain in OUTPUT INPUT; do
  n ip6tables -w -F "$chain"
  n ip6tables -w -A "$chain" -i lo -j ACCEPT 2>/dev/null || n ip6tables -w -A "$chain" -o lo -j ACCEPT
  n ip6tables -w -A "$chain" -j REJECT
done
"""


def bundle() -> bytes:
    """The proxy package: verifiers' egress proxy and policy matching, made standalone."""
    policy = inspect.getsource(runtime_config)
    proxy = inspect.getsource(egress).replace(
        "from verifiers.v1.configs.runtime import", "from vf_egress.policy import"
    )
    if "verifiers." in proxy.replace("verifiers:", "") or "verifiers." in policy:
        raise RuntimeError("the egress proxy must not import verifiers")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, text in (
            ("vf_egress/__init__.py", ""),
            ("vf_egress/policy.py", policy),
            ("vf_egress/egress.py", proxy),
            ("vf_egress/serve.py", SERVE),
        ):
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def framework_routes(routes: list[str]) -> list[str]:
    return [
        urlsplit(url)._replace(path="", query="", fragment="").geturl()
        for url in routes
    ]


class HostEgress:
    """Restrict one container on a Docker host VM to the execution policy.

    `start` serves the trusted-setup policy (open, private addresses included) before
    any setup command runs; `apply` installs the execution policy and cuts the
    container's namespace on its first restricted call.
    """

    def __init__(
        self,
        host: RuntimeBackend,
        container: str,
        config: NetworkPolicyConfig,
        peers: dict[str, str],
    ) -> None:
        self.host = host
        self.container = container
        self.config = config
        self.peers = peers
        """Compose service names by container ID."""
        self.root = f"{ROOT}/{uuid.uuid4().hex[:12]}"
        self.address = ""
        self.port = 0
        self.token = ""
        self.names: list[str] = []
        self._pid = ""
        self._gateways: list[str] = []
        self._subnets: list[str] = []
        self._hosts: list[str] = []
        self._cut = False

    async def _run(self, script: str, *args: str) -> str:
        result = await self.host.bind().run(
            ["sh", "-c", script, "vf-egress", *args], {}
        )
        if result.exit_code:
            raise SandboxError(
                f"workspace egress setup failed: {(result.stderr or result.stdout).strip()[-2000:]}"
            )
        return result.stdout

    async def _inspect(self, container: str) -> dict:
        return json.loads(
            await self._run('docker inspect --format "{{json .}}" "$1"', container)
        )

    async def start(self) -> None:
        missing = (
            await self._run(
                "for tool in " + " ".join(HOST_TOOLS) + '; do command -v "$tool" '
                '>/dev/null || printf "%s " "$tool"; done'
            )
        ).strip()
        if missing:
            raise SandboxError(
                f"workspace egress needs {missing} on the Compose host image"
            )
        state = await self._inspect(self.container)
        if not state["State"].get("Running"):
            raise SandboxError("workspace container is not running")
        self._pid = str(state["State"]["Pid"])
        networks = state["NetworkSettings"]["Networks"] or {}
        if not networks:
            raise SandboxError("workspace egress needs a Compose network")
        for name in sorted(networks):
            network = networks[name]
            if not network.get("Gateway") or not network.get("IPAddress"):
                raise SandboxError(f"workspace network {name!r} has no IPv4 gateway")
            self._gateways.append(network["Gateway"])
            self._subnets.append(
                str(
                    ip_interface(
                        f"{network['IPAddress']}/{network['IPPrefixLen']}"
                    ).network
                )
            )
        self.address = self._gateways[0]
        # Embedded DNS is cut; peers resolve through the workspace's hosts file.
        own = set(networks)
        for container, service in self.peers.items():
            if state["Id"].startswith(container):
                continue
            peer = await self._inspect(container)
            shared = [
                network["IPAddress"]
                for name, network in (peer["NetworkSettings"]["Networks"] or {}).items()
                if name in own and network.get("IPAddress")
            ]
            if shared:
                aliases = dict.fromkeys(
                    [
                        service,
                        *(
                            alias
                            for name in own
                            for alias in (
                                (peer["NetworkSettings"]["Networks"] or {})
                                .get(name, {})
                                .get("Aliases")
                                or []
                            )
                            if alias != container and not container.startswith(alias)
                        ),
                    ]
                )
                self._hosts.append(f"{shared[0]} {' '.join(aliases)}")
                self.names.extend(aliases)
        self._hosts_path = state["HostsPath"]
        versions = [
            f"{name}=={importlib.metadata.version(name)}" for name in REQUIREMENTS
        ]
        root = shlex.quote(self.root)
        await self.host.bind().write(f"{self.root}.tar.gz", bundle())
        await self._run(
            f"set -e; umask 077; mkdir -p {root}/pkg {root}/lib; "
            f"tar -xzf {root}.tar.gz -C {root}/pkg; rm -f {root}.tar.gz; "
            "python3 -m pip install --quiet --disable-pip-version-check "
            f'--no-warn-script-location --target {root}/lib "$@" '
            f"> {root}/pip.log 2>&1 || {{ tail -20 {root}/pip.log; exit 1; }}",
            *versions,
        )
        await self._write_policy(
            NetworkPolicyConfig(allow=["*"]), [], allow_non_global=True
        )
        await self.host.bind().run_background(
            [
                "python3",
                "-s",
                "-E",
                f"{self.root}/pkg/vf_egress/serve.py",
                "--lib",
                f"{self.root}/lib",
                "--host",
                self.address,
                "--policy",
                f"{self.root}/policy.json",
                "--info",
                f"{self.root}/info.json",
            ],
            {},
            f"{self.root}/proxy.log",
        )
        async with asyncio.timeout(60):
            while True:
                found = await self.host.bind().run(
                    ["cat", f"{self.root}/info.json"], {}
                )
                if found.exit_code == 0 and found.stdout.strip():
                    info = json.loads(found.stdout)
                    break
                log = await self.host.bind().run(
                    ["sh", "-c", f"tail -c 2000 {root}/proxy.log"], {}
                )
                if "Traceback" in log.stdout:
                    raise SandboxError(f"workspace egress proxy failed: {log.stdout}")
                await asyncio.sleep(0.5)
        self.port = int(info["port"])
        self.token = info["token"]

    async def _write_policy(
        self,
        config: NetworkPolicyConfig,
        routes: list[str],
        *,
        allow_non_global: bool,
    ) -> None:
        # Replace the file whole: the proxy rereads it for each new connection.
        data = json.dumps(
            {
                "allow": config.allow,
                "block": config.block,
                "routes": routes,
                "allow_non_global": allow_non_global,
            }
        ).encode()
        temporary = f"{self.root}/policy.{uuid.uuid4().hex}"
        await self.host.bind().write(temporary, data)
        await self._run(
            'chmod 600 "$1" && mv -f "$1" "$2"', temporary, f"{self.root}/policy.json"
        )

    def env(self) -> dict[str, str]:
        proxy = f"http://verifiers:{self.token}@{self.address}:{self.port}"
        exclusions = ",".join(["localhost", "127.0.0.1", *self.names])
        return {
            "HTTP_PROXY": proxy,
            "HTTPS_PROXY": proxy,
            "http_proxy": proxy,
            "https_proxy": proxy,
            "NO_PROXY": exclusions,
            "no_proxy": exclusions,
        }

    async def apply(self, routes: list[str] | None) -> None:
        """Serve `routes` plus the configured policy, then cut the namespace once.

        `None` reopens the proxy for trusted grading; the namespace stays cut, so
        traffic still leaves through the proxy.
        """
        if routes is None:
            await self._write_policy(NetworkPolicyConfig(), [], allow_non_global=True)
            return
        await self._write_policy(
            self.config, framework_routes(routes), allow_non_global=False
        )
        if self._cut:
            return
        if self._hosts:
            entries = "".join(f"{line}\n" for line in self._hosts)
            await self._run('printf "%s" "$2" >> "$1"', self._hosts_path, entries)
        await self._run(
            CUT,
            self._pid,
            self.address,
            str(self.port),
            " ".join(self._gateways),
            " ".join(self._subnets),
        )
        self._cut = True
