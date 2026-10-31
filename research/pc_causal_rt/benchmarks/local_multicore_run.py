"""Profile one local causal-RT workflow in an isolated worker process.

Notes
-----
The supplied workflow configuration controls native and transport thread counts.
CPU time and peak RSS include native input preparation, RT, diagnostics and
compiler children. Individual RT case wall times remain in the sealed summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    """Run the existing workflow and write a separate resource measurement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    if args.profile.exists():
        raise ValueError("Use a new profile filename for every invocation")
    config = args.config.resolve()
    cfg = json.loads(config.read_text())
    command = [
        sys.executable,
        "-m",
        "research.pc_causal_rt.workflow",
        "--root",
        str(args.root.resolve()),
        "--config",
        str(config),
    ]
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started = time.perf_counter()
    result = subprocess.run(command, check=False)
    elapsed = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    record = {
        "scope": "Entire workflow: native preparation, RT, diagnostics and compiler children",
        "started_utc": started_utc,
        "command": command,
        "configuration": cfg,
        "configuration_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "wall_seconds": elapsed,
        "user_cpu_seconds": after.ru_utime - before.ru_utime,
        "system_cpu_seconds": after.ru_stime - before.ru_stime,
        "peak_rss_KiB": after.ru_maxrss,
        "exit_code": result.returncode,
        "logical_cpus": os.cpu_count(),
        "cgroup_cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
        "cgroup_memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
    }
    args.profile.parent.mkdir(parents=True, exist_ok=True)
    args.profile.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
