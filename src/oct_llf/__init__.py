"""OCT log-probability scoring and calibration."""

from .calibration import (
    CALIBRATIONS,
    FORMULAS,
    TRAITS,
    WEIGHTINGS,
    align_score_to_behavior,
    calibrate,
    evaluation_row_block_permutation,
    off_diagonal_correlation,
    prompt_group_bootstrap,
    score_behavior_correlation,
)
from .scoring import (
    EncodedResponse,
    derived_score_arrays,
    extract_vllm_response_logprob,
    openrlhf_encode_response,
    score_formulas,
    stable_text_hash,
    validate_preference_pair,
)
from .validation import scan_repository_text, validate_bundle

__version__ = "0.1.0"

__all__ = [
    "CALIBRATIONS",
    "FORMULAS",
    "TRAITS",
    "WEIGHTINGS",
    "EncodedResponse",
    "align_score_to_behavior",
    "calibrate",
    "derived_score_arrays",
    "evaluation_row_block_permutation",
    "extract_vllm_response_logprob",
    "off_diagonal_correlation",
    "openrlhf_encode_response",
    "prompt_group_bootstrap",
    "scan_repository_text",
    "score_behavior_correlation",
    "score_formulas",
    "stable_text_hash",
    "validate_bundle",
    "validate_preference_pair",
]
