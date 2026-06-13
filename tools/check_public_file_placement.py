#!/usr/bin/env python

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".csv", ".gitkeep"}
SCRIPT_SUFFIXES = {".sh"}
HELPER_SUFFIXES = {".py"}


def iter_files(root):
    if not root.exists():
        return
    for path in root.rglob("*"):
        if "__pycache__" in path.parts:
            continue
        if path.is_file():
            if path.suffix in {".pyc", ".pyo"}:
                continue
            yield path


def is_allowed_report(path):
    if path.name == ".gitkeep":
        return True
    return path.suffix in REPORT_SUFFIXES


def main():
    problems = []

    reproduce_dir = REPO_ROOT / "reproduce"
    for path in iter_files(reproduce_dir):
        if not is_allowed_report(path):
            problems.append(f"Unexpected non-report file under reproduce/: {path.relative_to(REPO_ROOT)}")
        if path.suffix in HELPER_SUFFIXES | SCRIPT_SUFFIXES:
            problems.append(f"Misplaced helper/script under reproduce/: {path.relative_to(REPO_ROOT)}")

    for path in iter_files(REPO_ROOT / "tools"):
        if path.name == ".gitkeep":
            continue
        if path.suffix not in HELPER_SUFFIXES:
            problems.append(f"Unexpected non-Python helper under tools/: {path.relative_to(REPO_ROOT)}")

    for path in iter_files(REPO_ROOT / "scripts_autodl"):
        if path.name == ".gitkeep":
            continue
        if path.suffix not in SCRIPT_SUFFIXES:
            problems.append(f"Unexpected non-shell script under scripts_autodl/: {path.relative_to(REPO_ROOT)}")

    if problems:
        print("Public file placement check failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1

    print("Public file placement check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
