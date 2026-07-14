"""Run lightweight release hygiene checks."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
SKIP_TEXT_DIRS = {"dataset_metadata"}
TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".cff",
    ".gitignore",
}
MAX_FILE_MB = 50
ALLOWED_LARGE_FILES = {
    "baseline_models/clip/bpe_simple_vocab_16e6.txt.gz",
}

PROJECT_IDENTITY_PATTERNS = [
    re.compile("dume" + "ng98", re.IGNORECASE),
    re.compile("dumeng_" + r"1998@163\.com", re.IGNORECASE),
    re.compile("2607" + r"\.04607", re.IGNORECASE),
    re.compile("0000-0001-" + "6949-1125", re.IGNORECASE),
    re.compile(r"\b" + "Meng" + r"\s+" + "Du" + r"\b", re.IGNORECASE),
    re.compile(r"\b" + "Du" + r",\s*" + "Meng" + r"\b", re.IGNORECASE),
    re.compile(r"\b" + "Hongchang" + r"\s+" + "Chen" + r"\b", re.IGNORECASE),
    re.compile(r"\b" + "Chen" + r",\s*" + "Hongchang" + r"\b", re.IGNORECASE),
    re.compile(r"\b" + "Ran" + r"\s+" + "Li" + r"\b", re.IGNORECASE),
    re.compile(r"\b" + "Li" + r",\s*" + "Ran" + r"\b", re.IGNORECASE),
    re.compile(r"\b" + "Junjie" + r"\s+" + "Zhang" + r"\b", re.IGNORECASE),
    re.compile(r"\b" + "Zhang" + r",\s*" + "Junjie" + r"\b", re.IGNORECASE),
    re.compile(r"\b" + "Qi" + r"\s+" + "Ouyang" + r"\b", re.IGNORECASE),
    re.compile(r"\b" + "Ouyang" + r",\s*" + "Qi" + r"\b", re.IGNORECASE),
    re.compile(r"\b" + "Shuxin" + r"\s+" + "Liu" + r"\b", re.IGNORECASE),
    re.compile(r"\b" + "Liu" + r",\s*" + "Shuxin" + r"\b", re.IGNORECASE),
]

SENSITIVE_PATTERNS = [
    re.compile(r"192\.168\."),
    re.compile(r"BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY"),
    re.compile(r"(?i)\b(api[_-]?key|access[_-]?key|secret|password|passwd)\b\s*[:=]"),
    re.compile(r"(?i)\b(token)\b\s*[:=]\s*[A-Za-z0-9_\-]{12,}"),
    re.compile(r"(?<!\\)\b[A-Za-z]:[\\/]"),
    re.compile(r"/home/[A-Za-z0-9_.-]+"),
] + PROJECT_IDENTITY_PATTERNS


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        parts = set(path.relative_to(root).parts)
        if parts & SKIP_DIRS:
            continue
        yield path


def _check_large_files() -> List[str]:
    errors: List[str] = []
    for path in _iter_files(ROOT):
        rel = path.relative_to(ROOT).as_posix()
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_MB and rel not in ALLOWED_LARGE_FILES:
            errors.append(f"{rel} is {size_mb:.1f} MB")
    return errors


def _is_text_file(path: Path) -> bool:
    if path.name in {".gitignore"}:
        return True
    return path.suffix in TEXT_SUFFIXES


def _check_sensitive_text() -> List[Tuple[str, int, str]]:
    hits: List[Tuple[str, int, str]] = []
    for path in _iter_files(ROOT):
        rel_path = path.relative_to(ROOT)
        if set(rel_path.parts) & SKIP_TEXT_DIRS:
            continue
        if not _is_text_file(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern in SENSITIVE_PATTERNS:
                if pattern.search(line):
                    hits.append((rel_path.as_posix(), lineno, line.strip()))
    return hits


def _check_git_metadata() -> List[Tuple[str, int, str]]:
    if not (ROOT / ".git").exists():
        return []

    hits: List[Tuple[str, int, str]] = []
    commands = [
        (".git/remotes", ["git", "remote", "-v"]),
        (".git/history", ["git", "log", "--all", "--format=%an <%ae>"]),
    ]
    for label, command in commands:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        for lineno, line in enumerate(result.stdout.splitlines(), start=1):
            if any(pattern.search(line) for pattern in PROJECT_IDENTITY_PATTERNS):
                hits.append((label, lineno, line.strip()))
    return hits


def main() -> int:
    large_errors = _check_large_files()
    sensitive_hits = _check_sensitive_text() + _check_git_metadata()

    if not large_errors and not sensitive_hits:
        print("Release audit passed.")
        return 0

    if large_errors:
        print("Large-file warnings:")
        for item in large_errors:
            print(f"  - {item}")

    if sensitive_hits:
        print("Potential sensitive-text hits:")
        for rel, lineno, line in sensitive_hits:
            print(f"  - {rel}:{lineno}: {line}")

    return 1


if __name__ == "__main__":
    sys.exit(main())
