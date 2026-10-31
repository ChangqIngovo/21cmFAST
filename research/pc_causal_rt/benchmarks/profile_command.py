"""Profile a local scientific command without changing its runtime environment.

Notes
-----
CPU usage covers the command and descendants whose usage is propagated through
normal process waits. Linux ``ru_maxrss`` is the largest child's peak resident
set, not the simultaneous or summed memory of the process tree. The CLI starts
in a fresh process so earlier workloads do not contaminate this high-water mark.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import resource
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OMP_DYNAMIC",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "MKL_DYNAMIC",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def _limits() -> dict:
    """Record scheduler-visible limits without inferring aggregate memory use."""
    paths = (
        "/sys/fs/cgroup/cpu.max",
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory.high",
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    )
    cgroup = {}
    for name in paths:
        try:
            cgroup[name] = Path(name).read_text().strip()
        except OSError:
            cgroup[name] = None
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    return {
        "logical_cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "cgroup_raw_limits": cgroup,
    }


def profile_command(
    command: Sequence[str], output: Path, cwd: Path | None = None
) -> int:
    """Run a command, save resource measurements, and return its shell exit code.

    Parameters
    ----------
    command : sequence of str
        Exact executable and arguments. No shell expansion is performed.
    output : pathlib.Path
        New JSON file, created exclusively before the workload starts.
    cwd : pathlib.Path, optional
        Workload directory; defaults to the profiler's working directory.

    Returns
    -------
    int
        Workload exit code, or 128 plus its terminating signal. Launch failures
        return 127 for a missing executable and 126 for other operating errors.

    Notes
    -----
    Standard output/error and environment are inherited. A running placeholder
    is replaced in place with the final JSON record after the command returns.
    Existing output files are never opened for writing or reused.
    """
    if not command or any(not isinstance(arg, str) for arg in command):
        raise ValueError("A nonempty executable/argument list is required")
    output = Path(output)
    directory = Path.cwd() if cwd is None else Path(cwd).resolve()
    record = {
        "schema": 1,
        "status": "running",
        "command": list(command),
        "cwd": str(directory),
        "start_utc": _utc(),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "thread_scope": "Inherited environment; native input files may independently configure library threads",
        "resource_limits": _limits(),
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        started = time.perf_counter()
        try:
            result = subprocess.run(list(command), cwd=directory, check=False)
            returncode = result.returncode
            exit_code = returncode if returncode >= 0 else 128 - returncode
            record.update(
                status="finished" if returncode >= 0 else "signalled",
                workload_returncode=returncode,
                termination_signal=-returncode if returncode < 0 else None,
            )
        except OSError as error:
            exit_code = 127 if error.errno == errno.ENOENT else 126
            record.update(
                status="launch_failed",
                workload_returncode=None,
                launch_error=str(error),
                termination_signal=None,
            )
        elapsed = time.perf_counter() - started
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        # Linux reports KiB, Darwin bytes. Keep the raw value on other platforms.
        multiplier = (
            1024
            if sys.platform.startswith("linux")
            else 1
            if sys.platform == "darwin"
            else None
        )
        record.update(
            end_utc=_utc(),
            wall_seconds=elapsed,
            wrapper_exit_code=exit_code,
            descendant_user_cpu_seconds=after.ru_utime - before.ru_utime,
            descendant_system_cpu_seconds=after.ru_stime - before.ru_stime,
            cpu_scope="Wrapped command and waited-for descendants; background children not waited for are not included",
            memory={
                "largest_child_peak_rss_bytes": after.ru_maxrss * multiplier
                if multiplier
                else None,
                "ru_maxrss_raw": after.ru_maxrss,
                "ru_maxrss_raw_before_launch": before.ru_maxrss,
                "raw_units": "KiB"
                if multiplier == 1024
                else "bytes"
                if multiplier == 1
                else "platform-specific",
                "peak_rss_includes_prior_children": before.ru_maxrss != 0,
                "scope": "RUSAGE_CHILDREN largest-child peak RSS; not aggregate concurrent process-tree memory. Prior children, if present, also contribute to this high-water mark.",
            },
        )
        stream.seek(0)
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())
    return exit_code


def main() -> None:
    """Profile the command following ``--`` and preserve its completion status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Provide a workload command after --")
    try:
        exit_code = profile_command(command, args.output, args.cwd)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Profiler: {error}\n")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
