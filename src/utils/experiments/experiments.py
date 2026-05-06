#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any

import nbformat # type: ignore
from nbclient import NotebookClient # type: ignore
from nbclient.exceptions import CellExecutionError # type: ignore

CONFIG_CELL_PATTERN = re.compile(r"^\s*CONFIG\s*=\s*\{", re.MULTILINE)
ARTIFACT_LINE_PATTERN = re.compile(r"All artifacts in:\s*(.+)")
RUN_ID_PATTERN = re.compile(r"Run ID:\s*(\S+)")
METADATA_LINE_PATTERN = re.compile(r"^__EXPERIMENT_META__=(.+)$", re.MULTILINE)
ARTIFACT_MARKER_FILES = ("run_metadata.json", "global_metrics.json", "cv_results.json")


def configure_windows_event_loop_policy() -> None:
    if platform.system() != "Windows":
        return
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        # Keep execution resilient if policy cannot be changed in this environment.
        pass


def ensure_cell_ids(nb: nbformat.NotebookNode) -> None:
    for cell in nb.cells:
        if not cell.get("id"):
            cell["id"] = uuid.uuid4().hex[:8]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute a notebook once per config override from a JSON array."
    )
    parser.add_argument(
        "--notebook",
        default="lee2019_ssvep_finetune_sjepa_prelocal.ipynb",
        help="Path to the source notebook.",
    )
    parser.add_argument(
        "--configs",
        default="configurations.json",
        help="Path to a JSON file containing an array of CONFIG overrides.",
    )
    parser.add_argument(
        "--kernel-name",
        default=None,
        help="Optional Jupyter kernel name. Defaults to notebook metadata kernel if present.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=-1,
        help="Per-cell timeout in seconds. Use -1 for no timeout.",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run the sweep in a detached background process and exit immediately.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run id. Auto-generated when omitted.",
    )
    return parser.parse_args()


def find_config_cell_index(nb: nbformat.NotebookNode) -> int:
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        source = "".join(cell.source)
        if CONFIG_CELL_PATTERN.search(source):
            return idx
    raise RuntimeError("Could not find the CONFIG cell in the notebook.")


def make_override_cell(overrides: dict[str, Any]) -> nbformat.NotebookNode:
    source = (
        "# Injected by run_notebook_configs.py\n"
        f"_CONFIG_OVERRIDES = {pformat(overrides, width=100)}\n"
        "CONFIG.update(_CONFIG_OVERRIDES)\n"
        "print('Injected CONFIG overrides:')\n"
        "for _k in sorted(_CONFIG_OVERRIDES):\n"
        "    print(f'  {_k}: {CONFIG[_k]}')\n"
    )
    return nbformat.v4.new_code_cell(source=source)


def default_artifact_root(notebook_path: Path) -> Path:
    return notebook_path.parent.parent / "artifacts"


def create_run_id(configs: list[dict[str, Any]], configs_name: str) -> str:
    # Generate unique run ID from timestamp + configs hash + configs filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    config_str = json.dumps(configs, sort_keys=True, default=str)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    return f"{timestamp}_{configs_name}_{config_hash}"


def read_json_array(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Configurations file must contain a JSON array.")
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Configuration at index {i} is not a JSON object.")
    return data


def collect_text_outputs(nb: nbformat.NotebookNode) -> str:
    chunks: list[str] = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            output_type = output.get("output_type")
            if output_type == "stream":
                chunks.append(output.get("text", ""))
            elif output_type in {"execute_result", "display_data"}:
                text = output.get("data", {}).get("text/plain")
                if isinstance(text, list):
                    chunks.append("".join(text))
                elif isinstance(text, str):
                    chunks.append(text)
            elif output_type == "error":
                chunks.extend(output.get("traceback", []))
    return "\n".join(chunks)


def parse_artifact_info(output_text: str) -> tuple[str | None, str | None]:
    artifact_dir = None
    run_id = None

    artifact_matches = ARTIFACT_LINE_PATTERN.findall(output_text)
    if artifact_matches:
        artifact_dir = artifact_matches[-1].strip()

    run_id_matches = RUN_ID_PATTERN.findall(output_text)
    if run_id_matches:
        run_id = run_id_matches[-1].strip()

    return artifact_dir, run_id

def make_metadata_cell() -> nbformat.NotebookNode:
    source = (
        "# Injected by experiments.py (metadata capture)\n"
        "import json\n"
        "import sys\n"
        "_config = globals().get('CONFIG', {})\n"
        "_artifact_dir = (\n"
        "    globals().get('artifact_dir')\n"
        "    or globals().get('artifacts_dir')\n"
        "    or globals().get('ARTIFACT_DIR')\n"
        "    or globals().get('ARTIFACTS_DIR')\n"
        ")\n"
        "_run_id = (\n"
        "    globals().get('run_id')\n"
        "    or globals().get('RUN_ID')\n"
        ")\n"
        "if _artifact_dir is None and isinstance(_config, dict):\n"
        "    _artifact_dir = _config.get('artifact_dir') or _config.get('artifacts_dir')\n"
        "if _run_id is None and isinstance(_config, dict):\n"
        "    _run_id = _config.get('run_id')\n"
        "sys.stdout.write('__EXPERIMENT_META__=' + json.dumps({\n"
        "    'artifact_dir': str(_artifact_dir) if _artifact_dir is not None else None,\n"
        "    'run_id': str(_run_id) if _run_id is not None else None,\n"
        "}, default=str) + '\\n')\n"
        "sys.stdout.flush()\n"
    )
    return nbformat.v4.new_code_cell(source=source)


def parse_metadata_from_output(output_text: str) -> tuple[str | None, str | None]:
    matches = METADATA_LINE_PATTERN.findall(output_text)
    if not matches:
        return None, None

    try:
        payload = json.loads(matches[-1].strip())
    except json.JSONDecodeError:
        return None, None

    artifact_dir = payload.get("artifact_dir")
    run_id = payload.get("run_id")

    if isinstance(artifact_dir, str):
        artifact_dir = artifact_dir.strip() or None
    else:
        artifact_dir = None

    if isinstance(run_id, str):
        run_id = run_id.strip() or None
    else:
        run_id = None

    return artifact_dir, run_id


def is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False

    if platform.system() == "Windows":
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


def snapshot_artifact_dirs(artifact_root: Path) -> set[Path]:
    if not artifact_root.exists():
        return set()

    dirs: set[Path] = set()
    for marker_name in ARTIFACT_MARKER_FILES:
        for marker_path in artifact_root.rglob(marker_name):
            dirs.add(marker_path.parent.resolve())
    return dirs


def newest_recent_artifact_dir(
    artifact_dirs: set[Path],
    started_wall_time: float,
    recent_window_seconds: int = 120,
) -> str | None:
    candidates: list[tuple[float, Path]] = []
    for artifact_dir in artifact_dirs:
        marker_mtime = None
        for marker_name in ARTIFACT_MARKER_FILES:
            marker_path = artifact_dir / marker_name
            if marker_path.exists():
                marker_mtime = marker_path.stat().st_mtime
                break
        if marker_mtime is None:
            continue
        if marker_mtime >= started_wall_time - recent_window_seconds:
            candidates.append((marker_mtime, artifact_dir))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return str(candidates[0][1])


def write_status(status_path: Path, payload: dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_data = dict(payload)
    status_data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    status_path.write_text(json.dumps(status_data, indent=2))


def build_daemon_command(args: argparse.Namespace, run_id: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--notebook",
        str(args.notebook),
        "--configs",
        str(args.configs),
        "--timeout",
        str(args.timeout),
        "--run-id",
        run_id,
    ]
    if args.kernel_name:
        command.extend(["--kernel-name", args.kernel_name])
    return command


def launch_daemon(
    args: argparse.Namespace,
    run_id: str,
    notebook_path: Path,
    configs_path: Path,
    total_configs: int,
) -> int:
    results_dir = Path("experiment_results") / run_id
    status_path = results_dir / "status.json"
    log_path = results_dir / "daemon.log"
    pid_path = results_dir / "daemon.pid"

    command = build_daemon_command(args, run_id)
    results_dir.mkdir(parents=True, exist_ok=True)

    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(Path.cwd()),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    pid_path.write_text(f"{process.pid}\n")
    started_at = datetime.now().isoformat(timespec="seconds")

    write_status(
        status_path,
        {
            "run_id": run_id,
            "state": "daemonized",
            "pid": process.pid,
            "pid_running": is_pid_running(process.pid),
            "notebook": str(notebook_path),
            "configs": str(configs_path),
            "total_configs": total_configs,
            "completed_configs": 0,
            "failed_configs": 0,
            "results_dir": str(results_dir),
            "log_path": str(log_path),
            "started_at": started_at,
        },
    )

    print("Started daemonized experiment run.")
    print(f"Run ID:       {run_id}")
    print(f"PID:          {process.pid}")
    print(f"Results dir:  {results_dir}")
    print(f"Status file:  {status_path}")
    print(f"Daemon log:   {log_path}")
    return 0

def execute_one(
    base_nb: nbformat.NotebookNode,
    notebook_path: Path,
    overrides: dict[str, Any],
    kernel_name: str | None,
    timeout: int,
) -> dict[str, Any]:
    nb = deepcopy(base_nb)
    config_idx = find_config_cell_index(nb)
    nb.cells.insert(config_idx + 1, make_override_cell(overrides))
    nb.cells.append(make_metadata_cell())
    ensure_cell_ids(nb)

    artifact_root_value = (
        overrides.get("artifact_dir")
        or overrides.get("artifacts_dir")
        or default_artifact_root(notebook_path)
    )
    artifact_root = Path(artifact_root_value)
    before_dirs = snapshot_artifact_dirs(artifact_root)

    notebook_kwargs = {
        "nb": nb,
        "timeout": timeout,
        "kernel_name": kernel_name,
        "resources": {"metadata": {"path": str(notebook_path.parent)}},
    }

    client = NotebookClient(**notebook_kwargs)

    started = time.time()
    status = "completed"
    error_message = None

    try:
        client.execute()
    except CellExecutionError as exc:
        status = "failed"
        error_message = str(exc)

    elapsed_seconds = round(time.time() - started, 3)
    output_text = collect_text_outputs(nb)

    # 1) Prefer machine-readable metadata from the injected final cell.
    metadata_artifact_dir, metadata_run_id = parse_metadata_from_output(output_text)

    # 2) Fall back to your old human-readable log scraping.
    parsed_artifact_dir, parsed_run_id = parse_artifact_info(output_text)

    artifact_dir = metadata_artifact_dir or parsed_artifact_dir
    run_id = metadata_run_id or parsed_run_id

    # 3) Final fallback for artifact_dir: detect exactly one new directory.
    if artifact_dir is None and artifact_root.exists():
        after_dirs = snapshot_artifact_dirs(artifact_root)
        new_dirs = sorted(after_dirs - before_dirs)
        if len(new_dirs) == 1:
            artifact_dir = str(new_dirs[0])

    # 4) If no new directory appeared (e.g. re-run in same minute), pick the most recently
    # touched artifact directory based on marker files updated during this execution.
    if artifact_dir is None and artifact_root.exists():
        after_dirs = snapshot_artifact_dirs(artifact_root)
        artifact_dir = newest_recent_artifact_dir(after_dirs, started_wall_time=started)

    # 5) Optional fallback: if your artifact directory name is the run_id, recover it from there.
    if run_id is None and artifact_dir is not None:
        run_id = Path(artifact_dir).name

    return {
        "status": status,
        "error": error_message,
        "config": overrides,
        "artifact_dir": artifact_dir,
        "run_id": run_id,
        "elapsed_seconds": elapsed_seconds,
    }


def main() -> int:
    args = parse_args()
    configure_windows_event_loop_policy()
    notebook_path = Path(args.notebook).expanduser().resolve()
    configs_path = Path(args.configs).expanduser().resolve()

    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    if not configs_path.exists():
        raise FileNotFoundError(f"Configurations file not found: {configs_path}")

    configs = read_json_array(configs_path)
    run_id = args.run_id or create_run_id(configs, configs_path.stem)

    if args.daemon:
        return launch_daemon(
            args=args,
            run_id=run_id,
            notebook_path=notebook_path,
            configs_path=configs_path,
            total_configs=len(configs),
        )

    base_nb = nbformat.read(notebook_path, as_version=4)
    ensure_cell_ids(base_nb)

    results: list[dict[str, Any]] = []
    artifact_ids: list[str] = []
    started_at = datetime.now().isoformat(timespec="seconds")

    results_dir = Path("experiment_results") / run_id
    status_path = results_dir / "status.json"

    write_status(
        status_path,
        {
            "run_id": run_id,
            "state": "running",
            "pid": os.getpid(),
            "pid_running": True,
            "notebook": str(notebook_path),
            "configs": str(configs_path),
            "total_configs": len(configs),
            "completed_configs": 0,
            "failed_configs": 0,
            "results_dir": str(results_dir),
            "started_at": started_at,
        },
    )

    print(f"Notebook:       {notebook_path}")
    print(f"Configurations: {configs_path}")
    print(f"Total configs:  {len(configs)}")
    print()

    for idx, overrides in enumerate(configs, start=1):
        print(f"[{idx}/{len(configs)}] Running config override:")
        print(json.dumps(overrides, indent=2))
        result = execute_one(
            base_nb=base_nb,
            notebook_path=notebook_path,
            overrides=overrides,
            kernel_name=args.kernel_name,
            timeout=args.timeout,
        )
        results.append(result)

        if result["artifact_dir"]:
            artifact_ids.append(Path(result["artifact_dir"]).name)

        status = result["status"]
        print(f"  status:       {status}")
        print(f"  elapsed_sec:  {result['elapsed_seconds']}")
        print(f"  run_id:       {result['run_id']}")
        print(f"  artifact_dir: {result['artifact_dir']}")
        if result["error"]:
            print(f"  error:        {result['error']}")
        print()

        failed_count = sum(1 for item in results if item["status"] == "failed")
        write_status(
            status_path,
            {
                "run_id": run_id,
                "state": "running",
                "pid": os.getpid(),
                "pid_running": True,
                "notebook": str(notebook_path),
                "configs": str(configs_path),
                "total_configs": len(configs),
                "completed_configs": idx,
                "failed_configs": failed_count,
                "current_config_index": idx,
                "latest_result": result,
                "results_dir": str(results_dir),
                "started_at": started_at,
            },
        )

    # Write results into results/{run_id}/
    results_dir.mkdir(parents=True, exist_ok=True)
    configs_path = results_dir / "configs.json"
    results_path = results_dir / "results.json"
    artifact_ids_path = results_dir / "artifact_ids.json"

    configs_path.write_text(json.dumps(configs, indent=2))
    results_path.write_text(json.dumps(results, indent=2))
    artifact_ids_path.write_text(json.dumps(artifact_ids, indent=2))

    failed_count = sum(1 for item in results if item["status"] == "failed")
    final_state = "completed_with_failures" if failed_count else "completed"
    write_status(
        status_path,
        {
            "run_id": run_id,
            "state": final_state,
            "pid": os.getpid(),
            "pid_running": is_pid_running(os.getpid()),
            "notebook": str(notebook_path),
            "configs": str(configs_path),
            "total_configs": len(configs),
            "completed_configs": len(configs),
            "failed_configs": failed_count,
            "results_dir": str(results_dir),
            "summary_json": str(results_path),
            "artifact_ids_json": str(artifact_ids_path),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    print("Finished.")
    print(f"Results dir:    {results_dir}")
    print(f"Summary JSON:   {results_path}")
    print(f"Artifact list:  {artifact_ids_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
