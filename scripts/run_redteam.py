"""Run data-only attacker/victim scenarios in a disposable offline container.

No API keys, local configuration, cache, Git metadata or Docker socket are
mounted. Building the dependencies image may access package repositories;
attack execution uses --network none and a read-only source snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
IMAGE = "search-orchestrator-redteam:local"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Build the dependency image first")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=PROJECT, help="Application source snapshot to probe")
    args = parser.parse_args()
    cache = PROJECT / ".cache"
    cache.resolve().relative_to(PROJECT.resolve())
    cache.mkdir(exist_ok=True)
    docker = ["docker", "--config", str(cache / "docker-redteam")]
    with tempfile.TemporaryDirectory(prefix="redteam-", dir=cache) as temp:
        sandbox = Path(temp)
        sandbox.resolve().relative_to(cache.resolve())
        if args.build:
            context = sandbox / "build"
            context.mkdir()
            shutil.copy2(PROJECT / "scripts" / "redteam.Dockerfile", context / "Dockerfile")
            selected = ("ddgs", "httpcore", "httpx", "langchain-core", "langchain-openai", "langgraph", "lxml", "pydantic")
            requirements = [line for line in (PROJECT / "requirements.txt").read_text(encoding="utf-8").splitlines()
                            if line.split(">=", 1)[0] in selected]
            (context / "requirements-redteam.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
            subprocess.run(docker + ["build", "--tag", IMAGE, str(context)], check=True, timeout=900)

        snapshot = sandbox / "source"
        snapshot.mkdir()
        # Deliberate allowlist: never copy config*.json, .env, .git or caches.
        for name in ("security.py", "web_fetchers.py", "search_orchestrator.py", "evaluation_helpers.py",
                     "research_models.py", "source_assessment.py", "source_scoring.py", "app_settings.py"):
            shutil.copy2(args.source.resolve() / name, snapshot / name)
        shutil.copy2(PROJECT / "tests" / "redteam_scenarios.py", snapshot / "redteam_scenarios.py")
        container_name = "search-orchestrator-redteam-" + uuid.uuid4().hex[:12]
        command = docker + [
            "run", "--name", container_name, "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", "64", "--memory", "512m",
            "--cpus", "1", "--user", "65534:65534", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--mount", f"type=bind,source={snapshot},target=/app,readonly",
            IMAGE, "python", "-B", "/app/redteam_scenarios.py",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=120)
        finally:
            # Killing a timed-out Docker CLI does not reliably stop its
            # container. Explicitly remove only this run's unique container.
            subprocess.run(docker + ["rm", "--force", container_name], capture_output=True, timeout=30)
            remaining = subprocess.run(
                docker + ["ps", "--all", "--filter", f"name=^/{container_name}$", "--format", "{{.ID}}"],
                check=True, capture_output=True, text=True, timeout=30,
            )
            if remaining.stdout.strip():
                raise RuntimeError("The disposable red-team container could not be removed")
        if result.returncode not in {0, 1}:
            raise RuntimeError(f"Sandbox failed: {result.stderr[-2000:]}")
        report = json.loads(result.stdout)
        report["dependency_image_id"] = subprocess.run(
            docker + ["image", "inspect", "--format", "{{.Id}}", IMAGE],
            check=True, capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        report["source_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in sorted(snapshot.glob("*.py"))}
        report["sandbox"] = {
            "engine": "docker", "network": "none", "read_only": True, "uid": 65534,
            "memory_bytes": 512 * 1024 * 1024, "cpus": 1, "pids": 64,
            "source_mount": "allowlisted snapshot only; read-only", "container_removed": True,
            "model": "scripted compromised local-LLM role; no live LLM",
        }
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"passed": report["passed"], "failed": report["failed"],
                          "output": str(args.output.resolve())}, ensure_ascii=False))
        return int(report["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
