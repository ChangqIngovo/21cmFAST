"""Find an existing native runtime using only Python 3.6's standard library.

This bootstrap runs before the selected runtime's modules can be loaded. Keep
modern type expressions quoted and do not use future annotations here.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def _valid_path(path: Path) -> Path:
    if not path.is_absolute() or any(c in str(path) for c in ("\n", "\r", ",")):
        raise ValueError("Runtime and PBS paths must be absolute without commas or newlines")
    return path


def _read_pointer(path: Path) -> "Path | None":
    if not path.is_file():
        return None
    value = path.read_text().strip()
    return _valid_path(Path(value)) if value else None


def _ancestry(parent: Path) -> "list[Path]":
    chain = []
    path = parent.resolve()
    while True:
        if path in chain:
            raise ValueError("Cycle in runtime parent ancestry")
        chain.append(path)
        following = _read_pointer(path / "parent_run.txt")
        if following is None:
            following = _read_pointer(path / "runtime_parent.txt")
        if following is None:
            return chain
        path = following.resolve()


def _from_parent(parent: Path, python: "Path | None", runtime_env: "Path | None") -> dict:
    if not any((parent / marker).is_file() for marker in ("COMPLETED", "training/COMPLETED")):
        raise ValueError(f"Runtime parent is incomplete: {parent}")
    chain = _ancestry(parent)
    if python is None:
        native = [_read_pointer(path / "native_python.txt") for path in chain]
        python = next((path for path in native if path is not None), None)
    if python is None:
        for path in chain:
            manifest = path / "run_manifest.json"
            if manifest.is_file():
                info = json.loads(manifest.read_text())
                if info.get("python"):
                    python = _valid_path(Path(info["python"]))
                    break
    if python is None:
        candidates = [_read_pointer(path / "python_path.txt") for path in chain]
        candidates.extend(path / "venv/bin/python" for path in chain)
        python = next((path for path in candidates if path is not None and path.is_file()), None)
    if runtime_env is None:
        runtime_env = next((path / "runtime.env" for path in chain
                            if (path / "runtime.env").is_file()), None)
    if python is None or runtime_env is None:
        raise ValueError("No native runtime found in parent chain; set PC_PYTHON and PC_RUNTIME_ENV")
    return {"python": str(python), "parent": str(parent.resolve()), "runtime_env": str(runtime_env)}


def select_runtime(root: Path, *, parent: "Path | None" = None, python: "Path | None" = None,
                   runtime_env: "Path | None" = None) -> dict:
    """Find a completed native runtime, following training-run ancestry.

    Parameters
    ----------
    root : pathlib.Path
        Directory containing previous prepared runs.
    parent : pathlib.Path, optional
        Explicit completed parent run.
    python : pathlib.Path, optional
        Explicit native Python interpreter; overrides parent pointers.
    runtime_env : pathlib.Path, optional
        Shell file defining the environment module array ``PC_MODULES``.

    Returns
    -------
    dict
        Absolute interpreter, parent and module-file paths. Explicit local
        interpreters may use empty parent and module-file strings.
    """
    if python is not None:
        python = _valid_path(python.absolute())
    if runtime_env is not None:
        runtime_env = _valid_path(runtime_env.absolute())
    if parent is not None:
        result = _from_parent(_valid_path(parent.absolute()), python, runtime_env)
    elif python is not None:
        result = {"python": str(python), "parent": "", "runtime_env": str(runtime_env or "")}
    else:
        candidates = []
        for path in root.glob("*"):
            markers = [path / name for name in ("COMPLETED", "training/COMPLETED")
                       if (path / name).is_file()]
            if path.is_dir() and markers:
                candidates.append((max(marker.stat().st_mtime for marker in markers), path))
        errors = []
        for _, path in sorted(candidates, reverse=True):
            try:
                result = _from_parent(path, None, runtime_env)
                if not os.access(result["python"], os.X_OK):
                    raise ValueError("Python interpreter is missing or not executable")
                break
            except (OSError, ValueError) as error:
                errors.append(f"{path.name}: {error}")
        else:
            detail = "; ".join(errors[:3])
            raise ValueError(f"No completed native runtime under {root}; supply --parent or PC_PYTHON. {detail}")
    if not os.access(result["python"], os.X_OK):
        raise ValueError("Selected Python is missing or not executable: " + result["python"])
    if result["runtime_env"] and not Path(result["runtime_env"]).is_file():
        raise ValueError("Selected runtime.env is missing")
    return result


def main(argv: "list[str] | None" = None) -> None:
    """Print the selected interpreter, parent and environment-file paths.

    Parameters
    ----------
    argv : list of str, optional
        Command arguments; defaults to the current process arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select",))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--runtime-env", type=Path)
    args = parser.parse_args(argv)
    try:
        result = select_runtime(args.root, parent=args.parent, python=args.python,
                                runtime_env=args.runtime_env)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    sys.stdout.write("\n".join(result[name] for name in ("python", "parent", "runtime_env")) + "\n")


if __name__ == "__main__":
    main()
