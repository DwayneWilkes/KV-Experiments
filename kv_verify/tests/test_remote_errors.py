"""Tests for remote error handling (Task 7.6)."""

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from kv_verify.lib.remote import (
    RemoteConfig, RemoteSSHSession, RemoteError,
    sync_to_remote, sync_from_remote, run_remote_stage,
)


def _cfg():
    return RemoteConfig(backend="ssh", host="gpu.box", user="root",
                       key_path=Path("/tmp/key"), remote_dir="/workspace")


class TestConnectionFailure:
    @patch("subprocess.run")
    def test_ssh_connection_refused(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(255, "ssh", stderr="Connection refused")
        session = RemoteSSHSession(_cfg())
        with pytest.raises(RemoteError, match="SSH.*failed"):
            session.run_ssh("ls", check=True)


class TestRsyncFailure:
    @patch("subprocess.run")
    def test_rsync_failure_raises(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(12, "rsync", stderr="error in rsync protocol")
        session = RemoteSSHSession(_cfg())
        with pytest.raises(RemoteError, match="rsync.*failed"):
            session.run_rsync("/src/", "/dst/", upload=True, check=True)


class TestRemoteStageFailure:
    @patch("kv_verify.lib.remote.RemoteSSHSession.run_ssh")
    def test_nonzero_exit_does_not_raise(self, mock_ssh):
        """run_remote_stage uses check=False, so non-zero exits return, not raise."""
        mock_ssh.return_value = MagicMock(returncode=1, stdout="", stderr="CUDA OOM")
        # Should not raise — run_remote_stage passes check=False
        result = run_remote_stage(_cfg(), stage="extraction")
        # Verify run_ssh was called with a command containing the stage name
        call_args = str(mock_ssh.call_args)
        assert "extraction" in call_args

    @patch("kv_verify.lib.remote.RemoteSSHSession.run_ssh")
    def test_stage_command_includes_model_dir(self, mock_ssh):
        mock_ssh.return_value = MagicMock(returncode=0, stdout="", stderr="")
        run_remote_stage(_cfg(), stage="extraction")
        call_args = str(mock_ssh.call_args)
        assert "KV_VERIFY_MODEL_DIR" in call_args
