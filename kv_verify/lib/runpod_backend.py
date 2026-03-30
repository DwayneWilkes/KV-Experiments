"""RunPod backend for remote GPU execution.

Uses the RunPod REST API to create on-demand GPU pods, wait for them
to become available, extract SSH endpoints, and terminate when done.

Requires RUNPOD_API_KEY environment variable.

Usage:
    from kv_verify.lib.runpod_backend import RunPodSession

    session = RunPodSession(gpu_type="NVIDIA RTX 4090", api_key=os.environ["RUNPOD_API_KEY"])
    session.create(name="kv-verify", image="runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04")
    session.wait_for_ready()
    ssh_host, ssh_port = session.ssh_endpoint()
    # ... use RemoteSSHSession with these credentials ...
    session.terminate()

API reference: https://docs.runpod.io/pods/manage-pods
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

import requests

BASE_URL = "https://rest.runpod.io/v1"


@dataclass
class RunPodSession:
    """Manages a RunPod GPU pod lifecycle."""
    gpu_type: str
    api_key: str
    pod_id: Optional[str] = None
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def create(
        self,
        name: str = "kv-verify",
        image: str = "runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04",
        disk_gb: int = 20,
        volume_gb: int = 50,
    ) -> str:
        """Create an on-demand GPU pod. Returns pod ID."""
        payload = {
            "name": name,
            "imageName": image,
            "gpuTypeIds": [self.gpu_type],
            "gpuCount": 1,
            "containerDiskInGb": disk_gb,
            "volumeInGb": volume_gb,
            "ports": "22/tcp",  # SSH access
        }
        resp = requests.post(f"{BASE_URL}/pods", json=payload, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        self.pod_id = data["id"]
        return self.pod_id

    def get_status(self) -> dict:
        """Get current pod status and connection details."""
        resp = requests.get(f"{BASE_URL}/pods/{self.pod_id}", headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def wait_for_ready(self, timeout: int = 300, poll_interval: int = 5) -> None:
        """Poll until pod is RUNNING. Raises TimeoutError if not ready."""
        start = time.time()
        while time.time() - start < timeout:
            status = self.get_status()
            if status.get("desiredStatus") == "RUNNING" and status.get("runtime"):
                # Extract SSH connection details from runtime
                runtime = status["runtime"]
                ports = runtime.get("ports", [])
                for port in ports:
                    if port.get("privatePort") == 22:
                        self.ssh_host = port.get("ip")
                        self.ssh_port = port.get("publicPort")
                        return
            time.sleep(poll_interval)
        raise TimeoutError(f"Pod {self.pod_id} not ready after {timeout}s")

    def ssh_endpoint(self) -> tuple:
        """Return (host, port) for SSH connection."""
        if not self.ssh_host or not self.ssh_port:
            raise RuntimeError("Pod not ready. Call wait_for_ready() first.")
        return self.ssh_host, self.ssh_port

    def terminate(self) -> None:
        """Terminate the pod. Raises on failure to prevent orphaned billing."""
        if self.pod_id:
            resp = requests.delete(f"{BASE_URL}/pods/{self.pod_id}", headers=self._headers())
            resp.raise_for_status()
            self.pod_id = None
