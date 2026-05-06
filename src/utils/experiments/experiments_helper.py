#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_ROOT = Path("experiment_results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect status for notebook experiment runs."
    )
    parser.add_argument(
        "command",
        choices=["status", "running", "list", "stop"],
        help="status: show one run, running: show active runs, list: show all runs with status, stop: stop one run by PID from status.json",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID for the status command. If omitted, picks most recent run.",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root directory containing experiment result runs.",
    )
    return parser.parse_args()


def is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False

    if platform.system() == "Windows":
        # On Windows, os.kill(pid, 0) can raise WinError 87 for some PIDs.
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in result.stdout

    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_status(status_path: Path) -> dict[str, Any] | None:
    if not status_path.exists():
        return None
    try:
        payload = json.loads(status_path.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    pid = payload.get("pid")
    if isinstance(pid, int):
        payload["pid_running"] = is_pid_running(pid)
    return payload


def write_status(status_path: Path, payload: dict[str, Any]) -> None:
    status_path.write_text(json.dumps(payload, indent=2))


def find_run_dirs(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    return sorted([p for p in results_root.iterdir() if p.is_dir()], key=lambda p: p.name)


def print_status(run_id: str, status: dict[str, Any]) -> None:
    print(f"Run ID:            {run_id}")
    print(f"State:             {status.get('state')}")
    print(f"PID:               {status.get('pid')}")
    print(f"PID running:       {status.get('pid_running')}")
    print(f"Notebook:          {status.get('notebook')}")
    print(f"Configs:           {status.get('configs')}")
    print(f"Completed/Total:   {status.get('completed_configs')}/{status.get('total_configs')}")
    print(f"Failed configs:    {status.get('failed_configs')}")
    print(f"Updated at:        {status.get('updated_at')}")
    print(f"Results dir:       {status.get('results_dir')}")
    if status.get("log_path"):
        print(f"Daemon log:        {status.get('log_path')}")
    if status.get("summary_json"):
        print(f"Summary JSON:      {status.get('summary_json')}")


def cmd_status(results_root: Path, run_id: str | None) -> int:
    run_dirs = find_run_dirs(results_root)
    if not run_dirs:
        print(f"No runs found under: {results_root}")
        return 1

    target_dir: Path
    if run_id:
        target_dir = results_root / run_id
        if not target_dir.exists():
            print(f"Run not found: {run_id}")
            return 1
    else:
        target_dir = run_dirs[-1]

    status = read_status(target_dir / "status.json")
    if not status:
        print(f"No valid status file found for run: {target_dir.name}")
        return 1

    print_status(target_dir.name, status)
    return 0


def cmd_running(results_root: Path) -> int:
    run_dirs = find_run_dirs(results_root)
    active: list[tuple[str, dict[str, Any]]] = []

    for run_dir in run_dirs:
        status = read_status(run_dir / "status.json")
        if not status:
            continue
        if status.get("state") in {"running", "daemonized"} and status.get("pid_running"):
            active.append((run_dir.name, status))

    if not active:
        print("No active experiment runs found.")
        return 0

    print("Active experiment runs:")
    for run_id, status in active:
        print(
            f"- {run_id} | state={status.get('state')} | "
            f"progress={status.get('completed_configs')}/{status.get('total_configs')} | "
            f"pid={status.get('pid')}"
        )
    return 0


def cmd_list(results_root: Path) -> int:
    run_dirs = find_run_dirs(results_root)
    if not run_dirs:
        print(f"No runs found under: {results_root}")
        return 0

    print("Experiment runs:")
    for run_dir in run_dirs:
        status = read_status(run_dir / "status.json")
        if not status:
            print(f"- {run_dir.name} | status=missing")
            continue
        print(
            f"- {run_dir.name} | state={status.get('state')} | "
            f"progress={status.get('completed_configs')}/{status.get('total_configs')}"
        )
    return 0


def stop_pid(pid: int) -> tuple[bool, str]:
    if pid <= 0:
        return False, f"Invalid PID: {pid}"

    if platform.system() == "Windows":
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 and is_pid_running(pid):
            msg = result.stderr.strip() or result.stdout.strip() or "taskkill failed"
            return False, msg
        return True, result.stdout.strip() or "Process terminated."

    try:
        os.kill(pid, 15)
    except OSError as exc:
        if is_pid_running(pid):
            return False, str(exc)
        return True, "Process was already stopped."
    return True, "Process termination signal sent."


def cmd_stop(results_root: Path, run_id: str | None) -> int:
    run_dirs = find_run_dirs(results_root)
    if not run_dirs:
        print(f"No runs found under: {results_root}")
        return 1

    if run_id:
        target_dir = results_root / run_id
        if not target_dir.exists():
            print(f"Run not found: {run_id}")
            return 1
    else:
        target_dir = run_dirs[-1]

    status_path = target_dir / "status.json"
    status = read_status(status_path)
    if not status:
        print(f"No valid status file found for run: {target_dir.name}")
        return 1

    pid = status.get("pid")
    if not isinstance(pid, int):
        print(f"Run has no valid pid in status: {target_dir.name}")
        return 1

    if not is_pid_running(pid):
        print(f"PID {pid} is not running.")
        status["pid_running"] = False
        status["state"] = "stopped"
        status["updated_at"] = datetime.now().isoformat(timespec="seconds")
        write_status(status_path, status)
        return 0

    ok, message = stop_pid(pid)
    status["pid_running"] = False if ok else is_pid_running(pid)
    status["state"] = "stopped" if ok else status.get("state", "unknown")
    status["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_status(status_path, status)

    if ok:
        print(f"Stopped run {target_dir.name} (pid={pid}).")
        if message:
            print(message)
        return 0

    print(f"Failed to stop pid={pid} for run {target_dir.name}.")
    print(message)
    return 1


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root).expanduser().resolve()

    if args.command == "status":
        return cmd_status(results_root, args.run_id)
    if args.command == "running":
        return cmd_running(results_root)
    if args.command == "list":
        return cmd_list(results_root)
    if args.command == "stop":
        return cmd_stop(results_root, args.run_id)

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
