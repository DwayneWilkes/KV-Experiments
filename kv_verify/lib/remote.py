"""Remote GPU execution via SSH or RunPod.

Syncs kv_verify source and data to a remote machine, runs pipeline stages
via SSH, and syncs results back. RunPod wraps SSH with pod lifecycle management.

Usage:
    from kv_verify.lib.remote import RemoteConfig, sync_to_remote, sync_from_remote

    cfg = RemoteConfig.from_yaml("remote.yaml")
    sync_to_remote(cfg, local_source=Path("kv_verify"), output_dir=Path("output"))
    # ... run stages on remote ...
    sync_from_remote(cfg, local_output=Path("output"))
"""

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


class RemoteError(RuntimeError):
    """Error during remote execution (SSH, rsync, or pod lifecycle)."""
    pass
from typing import List, Optional

import yaml


@dataclass
class RemoteConfig:
    """Configuration for remote GPU execution."""
    backend: str = "ssh"  # "ssh" or "runpod"
    host: Optional[str] = None
    user: str = "root"
    key_path: Optional[Path] = None
    remote_dir: str = "/workspace/kv_verify"
    gpu_type: Optional[str] = None  # RunPod: "RTX_4090", "A100_40GB", etc.
    runpod_api_key: Optional[str] = None
    port: int = 22

    @classmethod
    def from_yaml(cls, path: Path) -> "RemoteConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if "key_path" in data and isinstance(data["key_path"], str):
            data["key_path"] = Path(data["key_path"])
        # API key from env var (never in YAML files)
        if "runpod_api_key" not in data or not data.get("runpod_api_key"):
            data["runpod_api_key"] = os.environ.get("RUNPOD_API_KEY")
        return cls(**data)


class RemoteSSHSession:
    """Execute commands and sync files via SSH/rsync."""

    def __init__(self, config: RemoteConfig):
        self.config = config

    def _ssh_base_args(self) -> List[str]:
        """Common SSH args: key, port, strict host checking off."""
        args = []
        if self.config.key_path:
            args.extend(["-i", str(self.config.key_path)])
        if self.config.port != 22:
            args.extend(["-p", str(self.config.port)])
        args.extend(["-o", "StrictHostKeyChecking=no"])
        return args

    def _ssh_cmd(self, command: str) -> List[str]:
        """Build a full SSH command."""
        target = f"{self.config.user}@{self.config.host}"
        return ["ssh"] + self._ssh_base_args() + [target, command]

    def _rsync_cmd(self, local: str, remote: str, upload: bool = True) -> List[str]:
        """Build an rsync command."""
        ssh_args = " ".join(self._ssh_base_args())
        target = f"{self.config.user}@{self.config.host}"
        cmd = ["rsync", "-avz", "--progress", "-e", f"ssh {ssh_args}"]
        if upload:
            cmd.extend([local, f"{target}:{remote}"])
        else:
            cmd.extend([f"{target}:{remote}", local])
        return cmd

    def run_ssh(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        """Execute a command on the remote machine."""
        cmd = self._ssh_cmd(command)
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=check)
        except subprocess.CalledProcessError as e:
            raise RemoteError(f"SSH command failed (exit {e.returncode}): {e.stderr}") from e

    def run_rsync(self, local: str, remote: str, upload: bool = True,
                  check: bool = True) -> subprocess.CompletedProcess:
        """Sync files between local and remote."""
        cmd = self._rsync_cmd(local, remote, upload=upload)
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=check)
        except subprocess.CalledProcessError as e:
            raise RemoteError(f"rsync failed (exit {e.returncode}): {e.stderr}") from e


def sync_to_remote(
    config: RemoteConfig,
    local_source: Path,
    output_dir: Path,
) -> None:
    """Upload kv_verify source and prompt data to remote."""
    session = RemoteSSHSession(config)
    remote_dir = config.remote_dir

    # Ensure remote dir exists
    session.run_ssh(f"mkdir -p {remote_dir}")

    # Sync source code
    session.run_rsync(f"{local_source}/", f"{remote_dir}/kv_verify/", upload=True)

    # Sync prompts/output if they exist
    prompts_dir = output_dir / "prompts"
    if prompts_dir.exists():
        session.run_rsync(f"{prompts_dir}/", f"{remote_dir}/output/prompts/", upload=True)


def sync_from_remote(
    config: RemoteConfig,
    local_output: Path,
) -> None:
    """Download results from remote back to local."""
    session = RemoteSSHSession(config)
    remote_dir = config.remote_dir

    local_output.mkdir(parents=True, exist_ok=True)

    # Sync features and cache
    for subdir in ("features", "cache"):
        session.run_rsync(
            f"{remote_dir}/output/{subdir}/",
            f"{local_output}/{subdir}/",
            upload=False,
            check=False,  # OK if dir doesn't exist yet
        )


def run_remote_stage(
    config: RemoteConfig,
    stage: str = "extraction",
    extra_args: Optional[List[str]] = None,
) -> subprocess.CompletedProcess:
    """Run a pipeline stage on the remote machine."""
    session = RemoteSSHSession(config)
    remote_dir = config.remote_dir

    cmd_parts = [
        f"cd {remote_dir}",
        f"KV_VERIFY_MODEL_DIR=/workspace/models",
        f"python -m kv_verify run --stages {stage}",
    ]
    if extra_args:
        cmd_parts[-1] += " " + " ".join(extra_args)

    full_cmd = " && ".join(cmd_parts[:1]) + " && " + " ".join(cmd_parts[1:])
    return session.run_ssh(full_cmd, check=False)
