"""Hashing, environment capture, and integrity manifests.

Everything that could change a Day-1 number has a recorded SHA-256: the
preregistration, the dataset, the prompt template, model provenance, raw score
files, analysis outputs, figures and the results document.

Nothing in this module reads, prints or stores credentials. Hugging Face tokens,
SSH keys and cluster secrets are never touched here.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

CHUNK_SIZE = 1 << 20


def repo_root() -> Path:
    """Repository root, resolved from this file's location."""
    return Path(__file__).resolve().parents[2]


def sha256_text(text: str) -> str:
    """SHA-256 hex digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    """SHA-256 hex digest of a file's contents."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialisation used for every checksummed document."""
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def sha256_json(obj: Any) -> str:
    """SHA-256 of the canonical JSON form of ``obj``."""
    return sha256_text(canonical_json(obj))


def write_json(path: str | Path, obj: Any) -> str:
    """Write ``obj`` as canonical JSON and return its SHA-256."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(obj)
    target.write_text(payload, encoding="utf-8")
    return sha256_text(payload)


def utc_now() -> str:
    """Current UTC timestamp in ISO-8601 form."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Source provenance
# --------------------------------------------------------------------------


def git_state(root: Path | None = None) -> dict[str, Any]:
    """Best-effort git commit / cleanliness record.

    Returns ``{"available": False, ...}`` rather than raising when git is absent
    or the tree is not a repository. Global git configuration is never modified.
    """
    root = root or repo_root()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "commit": None, "clean": None}
    return {
        "available": True,
        "commit": commit,
        "clean": status.strip() == "",
        "dirty_paths": sorted(
            line[3:] for line in status.splitlines() if line.strip()
        ),
    }


def source_manifest(paths: Iterable[str], root: Path | None = None) -> dict[str, Any]:
    """Per-file hashes plus a combined tree hash for the analysis code.

    A source manifest stands in for a git commit when git identity is not
    configured; it pins exactly the code that produced a result.
    """
    root = root or repo_root()
    files: dict[str, str | None] = {}
    for relative in sorted(paths):
        candidate = root / relative
        files[relative] = sha256_file(candidate) if candidate.is_file() else None
    combined = sha256_text(
        "\n".join(f"{name}:{digest}" for name, digest in sorted(files.items()))
    )
    return {"files": files, "tree_sha256": combined}


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # pragma: no cover - environment dependent
        return None


def environment_snapshot(include_gpu: bool = True) -> dict[str, Any]:
    """Interpreter, library and (optionally) GPU environment record."""
    snapshot: dict[str, Any] = {
        "captured_at": utc_now(),
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "packages": {
            name: _package_version(name)
            for name in (
                "numpy",
                "pandas",
                "pyarrow",
                "scipy",
                "matplotlib",
                "torch",
                "transformers",
                "tokenizers",
                "accelerate",
            )
        },
    }
    if include_gpu:
        snapshot["gpu"] = gpu_snapshot()
    return snapshot


def gpu_snapshot() -> dict[str, Any]:
    """CUDA/GPU record, or a reason why one is unavailable."""
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not installed"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "cuda not available"}
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "total_memory_bytes": int(props.total_memory),
                "capability": f"{props.major}.{props.minor}",
            }
        )
    return {
        "available": True,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "devices": devices,
    }


# --------------------------------------------------------------------------
# Integrity manifest
# --------------------------------------------------------------------------


def hash_paths(
    paths: Mapping[str, str], root: Path | None = None
) -> dict[str, dict[str, Any]]:
    """Hash a labelled collection of repository-relative paths."""
    root = root or repo_root()
    result: dict[str, dict[str, Any]] = {}
    for label, relative in paths.items():
        candidate = root / relative
        result[label] = {
            "path": relative,
            "present": candidate.is_file(),
            "sha256": sha256_file(candidate) if candidate.is_file() else None,
            "size_bytes": candidate.stat().st_size if candidate.is_file() else None,
        }
    return result


def hash_directory(
    relative_dir: str, pattern: str = "*", root: Path | None = None
) -> dict[str, Any]:
    """Hash every matching file in a directory, plus a combined digest."""
    root = root or repo_root()
    directory = root / relative_dir
    entries: dict[str, str] = {}
    if directory.is_dir():
        for path in sorted(directory.glob(pattern)):
            if path.is_file():
                entries[path.name] = sha256_file(path)
    combined = sha256_text(
        "\n".join(f"{name}:{digest}" for name, digest in sorted(entries.items()))
    )
    return {"path": relative_dir, "files": entries, "combined_sha256": combined}


# --------------------------------------------------------------------------
# Validators used by the readiness and integrity gates
# --------------------------------------------------------------------------

#: Fields every judge model must supply before it may be used for inference.
REQUIRED_MODEL_PROVENANCE_FIELDS: tuple[str, ...] = (
    "model_id",
    "revision",
    "tokenizer_revision",
    "dtype",
    "chat_template_sha256",
    "label_tokenization",
    "transformers_version",
    "torch_version",
)


def validate_model_provenance(record: Mapping[str, Any]) -> list[str]:
    """Return a list of problems with one model's provenance record.

    An empty list means the record is complete enough for inference. A revision
    must be a resolved commit-like string, not a branch name such as ``main``.
    """
    problems: list[str] = []
    for field_name in REQUIRED_MODEL_PROVENANCE_FIELDS:
        if record.get(field_name) in (None, "", {}, []):
            problems.append(f"missing field: {field_name}")
    revision = record.get("revision")
    if isinstance(revision, str):
        if revision in {"main", "master", "HEAD"}:
            problems.append(f"revision is unresolved: {revision!r}")
        elif len(revision) < 7:
            problems.append(f"revision does not look resolved: {revision!r}")
    tokenization = record.get("label_tokenization")
    if isinstance(tokenization, Mapping):
        if tokenization.get("scoring_method") in (None, ""):
            problems.append("label_tokenization is missing scoring_method")
        for key in ("met_token_ids", "not_met_token_ids"):
            if not tokenization.get(key):
                problems.append(f"label_tokenization is missing {key}")
    return problems


def write_checksum_file(path: str | Path, sha_path: str | Path) -> str:
    """Write a ``sha256sum``-style checksum file for ``path`` and return the digest."""
    target = Path(path)
    digest = sha256_file(target)
    Path(sha_path).write_text(f"{digest}  {target.name}\n", encoding="utf-8")
    return digest


def verify_checksum_file(path: str | Path, sha_path: str | Path) -> bool:
    """True when ``path`` still hashes to the digest recorded in ``sha_path``."""
    recorded = Path(sha_path).read_text(encoding="utf-8").split()[0]
    return recorded == sha256_file(path)
