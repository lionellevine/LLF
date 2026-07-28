#!/usr/bin/env python3
"""Score public OCT-style preference pairs with base and optional LoRA likelihoods.

The script imports Transformers and vLLM only after argument parsing, so ``--help``
works in the default CPU-only environment. Output is text-free JSONL containing
hashes, token counts, response-total log-probabilities, and derived formulas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from oct_llf.scoring import (
    EncodedResponse,
    extract_vllm_response_logprob,
    openrlhf_encode_response,
    score_formulas,
    stable_text_hash,
    validate_preference_pair,
)

INFERENCE_INSTALL = (
    "GPU scoring is optional. Install `oct-llf[inference]` plus the compatible "
    "torch and vLLM builds documented in README.md."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPU-score OCT chosen/rejected pairs to text-free JSONL."
    )
    parser.add_argument("input", type=Path, help="OCT-style JSONL with chosen/rejected lists")
    parser.add_argument("output", type=Path, help="text-free output JSONL")
    parser.add_argument("--model", required=True, help="base Hugging Face model name or path")
    parser.add_argument("--revision", required=True, help="immutable base model revision")
    parser.add_argument("--adapter", help="optional LoRA adapter local path or Hub repository")
    parser.add_argument("--adapter-revision", help="immutable Hub adapter revision")
    parser.add_argument("--adapter-name", default="oct-adapter")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser


def _inference_imports() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as exc:
        raise ImportError(INFERENCE_INSTALL) from exc
    return AutoTokenizer, LLM, SamplingParams, LoRARequest, snapshot_download


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            prompt, chosen, rejected = validate_preference_pair(
                row, row.get("row_id", line_number - 1)
            )
            rows.append(
                {
                    "row_id": row.get("row_id", len(rows)),
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "chosen_messages": row["chosen"],
                    "rejected_messages": row["rejected"],
                }
            )
    if not rows:
        raise ValueError("input contains no preference pairs")
    return rows


def _resolve_adapter(adapter: str, revision: str | None, snapshot_download: Any) -> str:
    path = Path(adapter)
    if path.exists():
        return str(path.resolve())
    if revision is None:
        raise ValueError("--adapter-revision is required for a Hub adapter")
    return str(snapshot_download(repo_id=adapter, revision=revision))


def _score_sequences(
    engine: Any,
    sampling: Any,
    encoded: list[EncodedResponse],
    batch_size: int,
    lora_request: Any | None,
) -> list[float]:
    totals: list[float] = []
    for start in range(0, len(encoded), batch_size):
        batch = encoded[start : start + batch_size]
        prompts = [{"prompt_token_ids": item.input_ids} for item in batch]
        outputs = engine.generate(
            prompts,
            sampling_params=sampling,
            lora_request=lora_request,
            use_tqdm=False,
        )
        if len(outputs) != len(batch):
            raise ValueError("vLLM returned a different number of outputs than inputs")
        totals.extend(
            extract_vllm_response_logprob(output, item)
            for output, item in zip(outputs, batch, strict=True)
        )
    return totals


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    AutoTokenizer, LLM, SamplingParams, LoRARequest, snapshot_download = _inference_imports()

    rows = _load_rows(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    chosen_encoded = [
        openrlhf_encode_response(tokenizer, row["chosen_messages"], args.max_length) for row in rows
    ]
    rejected_encoded = [
        openrlhf_encode_response(tokenizer, row["rejected_messages"], args.max_length)
        for row in rows
    ]
    encoded = chosen_encoded + rejected_encoded

    engine = LLM(
        model=args.model,
        revision=args.revision,
        tokenizer=args.model,
        tokenizer_revision=args.revision,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_lora=args.adapter is not None,
        max_loras=1,
        max_cpu_loras=1,
        max_model_len=args.max_length + 1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        dtype="bfloat16",
    )
    sampling = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    base_values = _score_sequences(engine, sampling, encoded, args.batch_size, None)

    adapter_values: list[float] | None = None
    if args.adapter:
        adapter_path = _resolve_adapter(args.adapter, args.adapter_revision, snapshot_download)
        request = LoRARequest(args.adapter_name, 1, adapter_path)
        adapter_values = _score_sequences(engine, sampling, encoded, args.batch_size, request)

    count = len(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            chosen_item = chosen_encoded[index]
            rejected_item = rejected_encoded[index]
            output: dict[str, Any] = {
                "row_id": row["row_id"],
                "prompt_hash": stable_text_hash(row["prompt"]),
                "chosen_hash": stable_text_hash(row["chosen"]),
                "rejected_hash": stable_text_hash(row["rejected"]),
                "prompt_tokens": chosen_item.prompt_length,
                "chosen_response_tokens": chosen_item.response_length,
                "rejected_response_tokens": rejected_item.response_length,
                "base_chosen_logprob": base_values[index],
                "base_rejected_logprob": base_values[count + index],
            }
            if adapter_values is not None:
                adapter_chosen = adapter_values[index]
                adapter_rejected = adapter_values[count + index]
                output.update(
                    {
                        "adapter_chosen_logprob": adapter_chosen,
                        "adapter_rejected_logprob": adapter_rejected,
                        **{
                            f"delta_{name}": value
                            for name, value in score_formulas(
                                adapter_chosen,
                                base_values[index],
                                chosen_item.response_length,
                                adapter_rejected,
                                base_values[count + index],
                                rejected_item.response_length,
                            ).items()
                        },
                    }
                )
            stream.write(json.dumps(output, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
