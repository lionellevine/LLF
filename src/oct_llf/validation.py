"""Integrity, schema, formula, metadata, and public-only validation."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .calibration import TRAITS
from .scoring import derived_score_arrays, sha256_file

SOURCE_DATA_REVISION = "2577813a6a435d21051c0548ff2f29dc897212d7"

FORBIDDEN_RAW_COLUMNS = frozenset(
    {
        "prompt",
        "chosen",
        "rejected",
        "messages",
        "text",
        "response",
        "completion",
        "raw_prompt",
        "raw_chosen",
        "raw_rejected",
        "prompt_text",
        "chosen_text",
        "rejected_text",
        "response_text",
        "completion_text",
        "raw_text",
    }
)
ALLOWED_STRING_COLUMNS = frozenset(
    {
        "dataset_trait",
        "source_revision",
        "prompt_hash",
        "chosen_hash",
        "rejected_hash",
    }
)
TEXT_SUFFIXES = frozenset(
    {
        ".md",
        ".txt",
        ".rst",
        ".py",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".csv",
        ".cff",
        ".gitignore",
    }
)


def _manifest_entries(manifest: Any) -> dict[str, dict[str, Any]]:
    if isinstance(manifest, list):
        entries = manifest
    elif isinstance(manifest, dict) and isinstance(manifest.get("datasets"), dict):
        entries = [dict(value, trait=key) for key, value in manifest["datasets"].items()]
    else:
        raise ValueError("manifest must be a list or contain a datasets mapping")
    result = {str(entry["trait"]): dict(entry) for entry in entries}
    if set(result) != set(TRAITS):
        raise ValueError("manifest must identify exactly the ten canonical traits")
    return result


def _metadata_has_raw_text(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower()
            if normalized == "contains_raw_text" and nested is not False:
                return True
            if normalized != "contains_raw_text" and (
                normalized in FORBIDDEN_RAW_COLUMNS
                or normalized.startswith("raw_")
                or normalized.endswith("_text")
            ):
                return True
            if _metadata_has_raw_text(nested):
                return True
    elif isinstance(value, list):
        return any(_metadata_has_raw_text(item) for item in value)
    return False


def validate_table(table: pa.Table, trait: str) -> dict[str, Any]:
    """Validate one public table and recompute every stored derived formula."""

    names = table.column_names
    forbidden = sorted(set(names) & FORBIDDEN_RAW_COLUMNS)
    if forbidden:
        raise ValueError(f"{trait}: forbidden raw-text columns: {forbidden}")
    string_columns = {field.name for field in table.schema if pa.types.is_string(field.type)}
    unexpected_strings = string_columns - ALLOWED_STRING_COLUMNS
    if unexpected_strings:
        raise ValueError(f"{trait}: unexpected string columns: {sorted(unexpected_strings)}")
    required = {
        "dataset_trait",
        "source_revision",
        "source_row_id",
        "prompt_hash",
        "chosen_hash",
        "rejected_hash",
        "prompt_tokens",
        "chosen_response_tokens",
        "rejected_response_tokens",
        "base_chosen_logprob",
        "base_rejected_logprob",
    }
    for adapter in TRAITS:
        required.update(
            {
                f"{adapter}_chosen_logprob",
                f"{adapter}_rejected_logprob",
                f"delta_sum_{adapter}",
                f"delta_lls_{adapter}",
                f"delta_separate_{adapter}",
            }
        )
    missing = required - set(names)
    unexpected = set(names) - required
    if missing:
        raise ValueError(f"{trait}: missing columns: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"{trait}: unexpected columns: {sorted(unexpected)}")
    if table.num_rows == 0:
        raise ValueError(f"{trait}: table is empty")
    dataset_values = set(table["dataset_trait"].to_pylist())
    if dataset_values != {trait}:
        raise ValueError(f"{trait}: dataset_trait values do not match filename")
    source_revisions = set(table["source_revision"].to_pylist())
    if source_revisions != {SOURCE_DATA_REVISION}:
        raise ValueError(f"{trait}: source_revision must be the pinned OCT dataset revision")
    if len(set(table["source_row_id"].to_pylist())) != table.num_rows:
        raise ValueError(f"{trait}: source_row_id values are not unique")
    for hash_name in ("prompt_hash", "chosen_hash", "rejected_hash"):
        hashes = table[hash_name].to_pylist()
        if any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in hashes
        ):
            raise ValueError(f"{trait}: invalid {hash_name} value")
    lengths = [
        np.asarray(table[name].to_numpy(), dtype=float)
        for name in ("prompt_tokens", "chosen_response_tokens", "rejected_response_tokens")
    ]
    if any(np.any(values <= 0) for values in lengths):
        raise ValueError(f"{trait}: token lengths must be positive")
    numeric_names = [field.name for field in table.schema if pa.types.is_floating(field.type)]
    if any(not np.all(np.isfinite(np.asarray(table[name].to_numpy()))) for name in numeric_names):
        raise ValueError(f"{trait}: nonfinite log-probability or formula value")
    metadata = table.schema.metadata or {}
    for key, raw_value in metadata.items():
        try:
            decoded: Any = json.loads(raw_value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            decoded = {
                "metadata_key": key.decode("utf-8", errors="replace"),
                "value": raw_value.hex(),
            }
        if _metadata_has_raw_text(decoded):
            raise ValueError(f"{trait}: schema metadata contains hidden raw-text fields")
    max_error = 0.0
    base_chosen = table["base_chosen_logprob"].to_numpy()
    base_rejected = table["base_rejected_logprob"].to_numpy()
    chosen_length = table["chosen_response_tokens"].to_numpy()
    rejected_length = table["rejected_response_tokens"].to_numpy()
    for adapter in TRAITS:
        derived = derived_score_arrays(
            table[f"{adapter}_chosen_logprob"].to_numpy(),
            base_chosen,
            chosen_length,
            table[f"{adapter}_rejected_logprob"].to_numpy(),
            base_rejected,
            rejected_length,
        )
        for formula, observed in derived.items():
            stored = np.asarray(table[f"delta_{formula}_{adapter}"].to_numpy(), dtype=float)
            error = float(np.max(np.abs(observed - stored)))
            max_error = max(max_error, error)
    if max_error != 0.0:
        raise ValueError(f"{trait}: formula round-trip maximum error is {max_error}, expected 0")
    return {"trait": trait, "rows": table.num_rows, "formula_round_trip_max_abs_error": max_error}


def scan_repository_text(root: str | Path) -> dict[str, Any]:
    """Scan repository text for internal paths, credentials, or signed URLs."""

    root = Path(root)
    literals = (
        "/" + "srv",
        "/" + "mnt",
        "artifact" + "://",
        "presigned" + "_url",
        "X-Amz-" + "Signature",
        "X-Goog-" + "Signature",
        "sig" + "nature=",
        "AWS_" + "SECRET_ACCESS_KEY",
        "OPENAI_" + "API_KEY",
        "ANTHROPIC_" + "API_KEY",
        "WANDB_" + "API_KEY",
        "GITHUB_" + "TOKEN",
        "GH_" + "TOKEN",
        "BEGIN " + "PRIVATE KEY",
        "<silico-" + "context>",
        "<silico-" + "hide>",
        "code_" + "export_handoff",
        "mcp__" + "silico",
    )
    path_patterns = (
        re.compile(r"/(?:home|tmp|opt|var|workspace)/[^\s`\"']+"),
        re.compile(r"[A-Za-z]:\\\\(?:Users|Windows)\\\\"),
        re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b"),
    )
    findings: list[dict[str, Any]] = []
    checked = 0
    ignored_parts = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".venv",
        ".ruff_cache",
        ".pi-subagents",
        ".claude",
        ".idea",
        ".vscode",
        "slurm-logs",
        "__pycache__",
        "build",
        "dist",
    }
    forbidden_binary_suffixes = {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}
    ignored_files = {
        ".bash_profile",
        ".bashrc",
        ".gitconfig",
        ".gitmodules",
        ".mcp.json",
        ".profile",
        ".ripgreprc",
        ".silico-session",
        ".silico-session.json",
        ".zprofile",
        ".zshrc",
    }
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or ignored_parts.intersection(path.parts)
            or path.name in ignored_files
        ):
            continue
        if path.stat().st_size >= 100_000_000:
            findings.append(
                {
                    "file": str(path.relative_to(root)),
                    "line": None,
                    "pattern": "file_size_at_least_100mb",
                }
            )
        if path.suffix.lower() in forbidden_binary_suffixes:
            findings.append(
                {
                    "file": str(path.relative_to(root)),
                    "line": None,
                    "pattern": "model_or_checkpoint_binary",
                }
            )
        if path.suffix == ".parquet":
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            "LICENSE",
            "NOTICE-DATA",
            ".gitignore",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        checked += 1
        for line_number, line in enumerate(text.splitlines(), 1):
            for literal in literals:
                if literal.lower() in line.lower():
                    findings.append(
                        {
                            "file": str(path.relative_to(root)),
                            "line": line_number,
                            "pattern": literal,
                        }
                    )
            if path.name != "validation.py":
                for pattern in path_patterns:
                    if pattern.search(line):
                        findings.append(
                            {
                                "file": str(path.relative_to(root)),
                                "line": line_number,
                                "pattern": "absolute_path",
                            }
                        )
    return {"passed": not findings, "files_checked": checked, "findings": findings}


def validate_bundle(
    data_dir: str | Path,
    manifest_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate all ten immutable bundle files against manifest and schema invariants."""

    data_root = Path(data_dir)
    parquet_root = data_root / "logprobs" if (data_root / "logprobs").is_dir() else data_root
    manifest_file = Path(manifest_path) if manifest_path else data_root / "bundle_manifest.json"
    entries = _manifest_entries(json.loads(manifest_file.read_text()))
    details: list[dict[str, Any]] = []
    errors: list[str] = []
    reference_schema: pa.Schema | None = None
    total_rows = total_bytes = 0
    max_error = 0.0
    for trait in TRAITS:
        path = parquet_root / f"{trait}.parquet"
        entry = entries[trait]
        try:
            if entry.get("contains_raw_text") is not False:
                raise ValueError("manifest must explicitly declare contains_raw_text=false")
            if float(entry.get("formula_round_trip_max_abs_error", float("nan"))) != 0.0:
                raise ValueError("manifest formula_round_trip_max_abs_error must be zero")
            if not path.is_file():
                raise ValueError("file is missing")
            size = path.stat().st_size
            digest = sha256_file(path)
            if size != int(entry["size_bytes"]):
                raise ValueError(f"size {size} != manifest {entry['size_bytes']}")
            if digest != entry["sha256"]:
                raise ValueError(f"SHA-256 {digest} != manifest {entry['sha256']}")
            table = pq.read_table(path)
            if table.num_rows != int(entry["rows"]):
                raise ValueError(f"rows {table.num_rows} != manifest {entry['rows']}")
            schema_without_metadata = table.schema.remove_metadata()
            if reference_schema is None:
                reference_schema = schema_without_metadata
            elif not schema_without_metadata.equals(reference_schema):
                raise ValueError("schema differs from the first dataset")
            detail = validate_table(table, trait)
            detail.update({"size_bytes": size, "sha256": digest, "passed": True})
            total_rows += table.num_rows
            total_bytes += size
            max_error = max(max_error, detail["formula_round_trip_max_abs_error"])
            details.append(detail)
        except Exception as exc:
            errors.append(f"{trait}: {exc}")
            details.append({"trait": trait, "passed": False, "error": str(exc)})
    scan = scan_repository_text(repository_root) if repository_root is not None else None
    if scan is not None and not scan["passed"]:
        errors.append("repository public-only scan failed")
    return {
        "passed": not errors,
        "datasets_validated": sum(bool(item.get("passed")) for item in details),
        "total_rows": total_rows,
        "total_bytes": total_bytes,
        "formula_round_trip_max_abs_error": max_error,
        "schema_identical": not any("schema differs" in error for error in errors),
        "public_only": scan,
        "errors": errors,
        "datasets": details,
    }
