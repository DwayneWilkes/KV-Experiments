"""Tests for RunPod backend (Tasks 8.1-8.2)."""

from unittest.mock import patch, MagicMock

import pytest

from kv_verify.lib.runpod_backend import RunPodSession


class TestRunPodSession:

    def test_create_pod_sends_request(self):
        with patch("kv_verify.lib.runpod_backend.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200, json=lambda: {"id": "pod-abc"})
            session = RunPodSession(gpu_type="NVIDIA RTX 4090", api_key="test-key")
            pod_id = session.create()
            assert pod_id == "pod-abc"
            # Verify the request was sent with correct GPU type
            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert "NVIDIA RTX 4090" in payload["gpuTypeIds"]

    def test_wait_for_ready_extracts_ssh(self):
        with patch("kv_verify.lib.runpod_backend.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, json=lambda: {
                "id": "pod-abc",
                "desiredStatus": "RUNNING",
                "runtime": {
                    "ports": [{"privatePort": 22, "ip": "194.68.1.1", "publicPort": 43210}]
                },
            })
            session = RunPodSession(gpu_type="NVIDIA RTX 4090", api_key="test-key")
            session.pod_id = "pod-abc"
            session.wait_for_ready(poll_interval=0)
            host, port = session.ssh_endpoint()
            assert host == "194.68.1.1"
            assert port == 43210

    def test_terminate_sends_delete(self):
        with patch("kv_verify.lib.runpod_backend.requests.delete") as mock_del:
            session = RunPodSession(gpu_type="NVIDIA RTX 4090", api_key="test-key")
            session.pod_id = "pod-abc"
            session.terminate()
            mock_del.assert_called_once()
            assert session.pod_id is None

    def test_ssh_endpoint_raises_when_not_ready(self):
        session = RunPodSession(gpu_type="NVIDIA RTX 4090", api_key="test-key")
        with pytest.raises(RuntimeError, match="not ready"):
            session.ssh_endpoint()

    def test_timeout_raises(self):
        with patch("kv_verify.lib.runpod_backend.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, json=lambda: {
                "id": "pod-abc", "desiredStatus": "STARTING", "runtime": None,
            })
            session = RunPodSession(gpu_type="NVIDIA RTX 4090", api_key="test-key")
            session.pod_id = "pod-abc"
            with pytest.raises(TimeoutError):
                session.wait_for_ready(timeout=1, poll_interval=0.1)
