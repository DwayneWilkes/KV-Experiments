"""Tests for remote execution module (Tasks 7.1-7.6)."""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from kv_verify.lib.remote import (
    RemoteConfig,
    RemoteSSHSession,
    sync_from_remote,
    sync_to_remote,
)


class TestRemoteConfig:
    """Task 7.1: RemoteConfig dataclass."""

    def test_ssh_config(self):
        cfg = RemoteConfig(
            backend="ssh",
            host="gpu.example.com",
            user="ubuntu",
            key_path=Path("~/.ssh/id_rsa"),
            remote_dir="/workspace/kv_verify",
        )
        assert cfg.backend == "ssh"
        assert cfg.host == "gpu.example.com"

    def test_runpod_config(self):
        cfg = RemoteConfig(
            backend="runpod",
            gpu_type="RTX_4090",
            remote_dir="/workspace/kv_verify",
        )
        assert cfg.backend == "runpod"

    def test_from_yaml(self, tmp_path):
        yaml_data = {
            "backend": "ssh",
            "host": "10.0.0.1",
            "user": "root",
            "key_path": "/root/.ssh/id_rsa",
            "remote_dir": "/workspace",
        }
        path = tmp_path / "remote.yaml"
        path.write_text(yaml.dump(yaml_data))
        cfg = RemoteConfig.from_yaml(path)
        assert cfg.host == "10.0.0.1"
        assert cfg.remote_dir == "/workspace"

    def test_api_key_from_env(self, tmp_path):
        yaml_data = {"backend": "runpod", "remote_dir": "/workspace"}
        path = tmp_path / "remote.yaml"
        path.write_text(yaml.dump(yaml_data))
        with patch.dict(os.environ, {"RUNPOD_API_KEY": "test-key-123"}):
            cfg = RemoteConfig.from_yaml(path)
            assert cfg.runpod_api_key == "test-key-123"


class TestRemoteSSHSession:
    """Task 7.2: SSH session via subprocess."""

    def test_constructs_ssh_command(self):
        cfg = RemoteConfig(backend="ssh", host="gpu.box", user="ubuntu",
                          key_path=Path("/tmp/key"), remote_dir="/workspace")
        session = RemoteSSHSession(cfg)
        cmd = session._ssh_cmd("ls /workspace")
        assert "ssh" in cmd[0]
        assert "ubuntu@gpu.box" in " ".join(cmd)
        assert "ls /workspace" in " ".join(cmd)

    def test_constructs_rsync_upload_command(self):
        cfg = RemoteConfig(backend="ssh", host="gpu.box", user="ubuntu",
                          key_path=Path("/tmp/key"), remote_dir="/workspace")
        session = RemoteSSHSession(cfg)
        cmd = session._rsync_cmd("/local/src/", "/workspace/src/", upload=True)
        assert "rsync" in cmd[0]
        assert "/local/src/" in " ".join(cmd)
        assert "ubuntu@gpu.box:/workspace/src/" in " ".join(cmd)

    def test_constructs_rsync_download_command(self):
        cfg = RemoteConfig(backend="ssh", host="gpu.box", user="ubuntu",
                          key_path=Path("/tmp/key"), remote_dir="/workspace")
        session = RemoteSSHSession(cfg)
        # _rsync_cmd(local, remote, upload=False): remote -> local
        cmd = session._rsync_cmd("/tmp/local_output/", "/workspace/output/", upload=False)
        cmd_str = " ".join(cmd)
        assert "ubuntu@gpu.box:/workspace/output/" in cmd_str
        assert "/tmp/local_output/" in cmd_str


class TestSyncFunctions:
    """Tasks 7.3-7.4: sync_to_remote and sync_from_remote."""

    @patch("kv_verify.lib.remote.RemoteSSHSession.run_ssh")
    @patch("kv_verify.lib.remote.RemoteSSHSession.run_rsync")
    def test_sync_to_remote_includes_source(self, mock_rsync, mock_ssh, tmp_path):
        mock_rsync.return_value = None
        mock_ssh.return_value = None
        cfg = RemoteConfig(backend="ssh", host="gpu.box", user="root",
                          key_path=Path("/tmp/key"), remote_dir="/workspace")
        sync_to_remote(cfg, local_source=tmp_path / "kv_verify", output_dir=tmp_path / "output")
        assert mock_rsync.called

    @patch("kv_verify.lib.remote.RemoteSSHSession.run_rsync")
    def test_sync_from_remote_downloads_output(self, mock_rsync, tmp_path):
        mock_rsync.return_value = None
        cfg = RemoteConfig(backend="ssh", host="gpu.box", user="root",
                          key_path=Path("/tmp/key"), remote_dir="/workspace")
        sync_from_remote(cfg, local_output=tmp_path / "output")
        assert mock_rsync.called
