"""Pure helpers for OCT preference-pair encoding and log-probability scoring."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EncodedResponse:
    """A rendered response and the token boundary used for response-only scoring."""

    input_ids: list[int]
    prompt_length: int
    response_length: int
    full_text: str


def stable_text_hash(text: str) -> str:
    """Return a stable UTF-8 SHA-256 identity hash."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_preference_pair(
    row: Mapping[str, Any], row_id: int | str = "?"
) -> tuple[str, str, str]:
    """Validate one two-message chosen/rejected pair and return its three texts."""

    chosen, rejected = row.get("chosen"), row.get("rejected")
    if not isinstance(chosen, list) or not isinstance(rejected, list):
        raise ValueError(f"row {row_id}: chosen/rejected must be message lists")
    if len(chosen) != 2 or len(rejected) != 2:
        raise ValueError(f"row {row_id}: expected one user and one assistant message per response")
    roles = ["user", "assistant"]
    if [item.get("role") for item in chosen] != roles or [
        item.get("role") for item in rejected
    ] != roles:
        raise ValueError(f"row {row_id}: each response must have user then assistant roles")
    prompt = chosen[0].get("content")
    chosen_text = chosen[1].get("content")
    rejected_text = rejected[1].get("content")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"row {row_id}: prompt is empty or non-string")
    if rejected[0].get("content") != prompt:
        raise ValueError(f"row {row_id}: chosen and rejected prompts differ")
    if not isinstance(chosen_text, str) or not chosen_text:
        raise ValueError(f"row {row_id}: chosen response is empty or non-string")
    if not isinstance(rejected_text, str) or not rejected_text:
        raise ValueError(f"row {row_id}: rejected response is empty or non-string")
    return prompt, chosen_text, rejected_text


def openrlhf_encode_response(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    max_length: int = 1024,
) -> EncodedResponse:
    """Encode a response compatibly with OpenRLHF RewardDataset at ``d21a99…``.

    The response boundary is established by rendering the prompt with a generation
    marker, then slicing the full chat rendering by character length. Sequences are
    tokenized without special-token insertion. An EOS is appended if absent and its
    final token ID is forced to the tokenizer EOS ID. This function deliberately
    fails rather than truncating, because clipping would silently change the score.
    """

    if max_length < 3:
        raise ValueError("max_length must be at least 3")
    prompt_messages = list(messages[:-1])
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_render = tokenizer.apply_chat_template(list(messages), tokenize=False)
    if not isinstance(prompt_text, str) or not isinstance(full_render, str):
        raise TypeError("chat template must return strings when tokenize=False")
    if not full_render.startswith(prompt_text):
        raise ValueError("full chat rendering does not start with the generation prompt")
    response_text = full_render[len(prompt_text) :]
    full_text = (prompt_text + response_text).rstrip("\n")
    eos_token = getattr(tokenizer, "eos_token", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token is None or eos_token_id is None:
        raise ValueError("tokenizer must define eos_token and eos_token_id")
    if not full_text.endswith(eos_token):
        full_text += " " + eos_token
    kwargs = {"padding": False, "truncation": False, "add_special_tokens": False}
    prompt_ids = list(tokenizer(prompt_text, **kwargs)["input_ids"])
    input_ids = list(tokenizer(full_text, **kwargs)["input_ids"])
    if not input_ids:
        raise ValueError("tokenized response is empty")
    if len(prompt_ids) >= max_length - 2:
        raise ValueError(f"prompt length {len(prompt_ids)} leaves fewer than two response tokens")
    if len(input_ids) > max_length:
        raise ValueError(
            f"untruncated sequence length {len(input_ids)} exceeds configured maximum {max_length}"
        )
    input_ids[-1] = int(eos_token_id)
    response_length = len(input_ids) - len(prompt_ids)
    if response_length <= 0:
        raise ValueError("response token boundary is empty")
    return EncodedResponse(input_ids, len(prompt_ids), response_length, full_text)


def extract_vllm_response_logprob(output: Any, encoded: EncodedResponse) -> float:
    """Sum actual-token vLLM prompt log-probabilities over response tokens only."""

    prompt_logprobs = getattr(output, "prompt_logprobs", None)
    if prompt_logprobs is None or len(prompt_logprobs) != len(encoded.input_ids):
        raise ValueError("vLLM prompt log-probabilities do not align with input token IDs")
    total = 0.0
    for position in range(encoded.prompt_length, len(encoded.input_ids)):
        token_id = encoded.input_ids[position]
        candidates = prompt_logprobs[position]
        if candidates is None or token_id not in candidates:
            raise ValueError(f"vLLM omitted actual token {token_id} at position {position}")
        value = candidates[token_id]
        logprob = value.logprob if hasattr(value, "logprob") else value
        total += float(logprob)
    if not math.isfinite(total):
        raise ValueError("response log-probability is nonfinite")
    return total


def score_formulas(
    adapter_chosen: float,
    base_chosen: float,
    chosen_length: int,
    adapter_rejected: float,
    base_rejected: float,
    rejected_length: int,
) -> dict[str, float]:
    """Compute unnormalized sum, combined-length (LLS), and separate-response scores."""

    if chosen_length <= 0 or rejected_length <= 0:
        raise ValueError("response lengths must be positive")
    values = (adapter_chosen, base_chosen, adapter_rejected, base_rejected)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("log-probabilities must be finite")
    chosen_shift = float(adapter_chosen) - float(base_chosen)
    rejected_shift = float(adapter_rejected) - float(base_rejected)
    delta_sum = chosen_shift - rejected_shift
    return {
        "sum": delta_sum,
        "lls": delta_sum / (chosen_length + rejected_length),
        "separate": chosen_shift / chosen_length - rejected_shift / rejected_length,
    }


def derived_score_arrays(
    adapter_chosen: Sequence[float] | np.ndarray,
    base_chosen: Sequence[float] | np.ndarray,
    chosen_length: Sequence[int] | np.ndarray,
    adapter_rejected: Sequence[float] | np.ndarray,
    base_rejected: Sequence[float] | np.ndarray,
    rejected_length: Sequence[int] | np.ndarray,
) -> dict[str, np.ndarray]:
    """Vectorized counterpart of :func:`score_formulas`."""

    ac, bc, ar, br = (
        np.asarray(value, dtype=float)
        for value in (adapter_chosen, base_chosen, adapter_rejected, base_rejected)
    )
    cl, rl = np.asarray(chosen_length, dtype=float), np.asarray(rejected_length, dtype=float)
    arrays = (ac, bc, cl, ar, br, rl)
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("all score arrays must have the same shape")
    if np.any(cl <= 0) or np.any(rl <= 0):
        raise ValueError("response lengths must be positive")
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("score inputs must be finite")
    chosen_shift, rejected_shift = ac - bc, ar - br
    summed = chosen_shift - rejected_shift
    return {
        "sum": summed,
        "lls": summed / (cl + rl),
        "separate": chosen_shift / cl - rejected_shift / rl,
    }
