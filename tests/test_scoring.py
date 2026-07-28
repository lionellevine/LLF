from __future__ import annotations

import unittest

from oct_llf.scoring import (
    EncodedResponse,
    derived_score_arrays,
    extract_vllm_response_logprob,
    openrlhf_encode_response,
    score_formulas,
    stable_text_hash,
    validate_preference_pair,
)


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 999

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        del tokenize
        if add_generation_prompt:
            return "USER:x\nASSISTANT:"
        return "USER:x\nASSISTANT:" + messages[-1]["content"] + "<eos>\n"

    def __call__(self, text, **kwargs):
        self.last_kwargs = kwargs
        return {"input_ids": list(range(len(text)))}


class ScoringTests(unittest.TestCase):
    def test_formulas_and_vectorized_equivalence(self):
        result = score_formulas(-8, -10, 4, -10, -13, 3)
        self.assertEqual(result["sum"], -1.0)
        self.assertAlmostEqual(result["lls"], -1 / 7)
        self.assertAlmostEqual(result["separate"], -0.5)
        arrays = derived_score_arrays([-8], [-10], [4], [-10], [-13], [3])
        for name, value in result.items():
            self.assertEqual(float(arrays[name][0]), value)

    def test_positive_lengths_are_required(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            score_formulas(1, 1, 0, 1, 1, 2)

    def test_pair_validation_and_hash(self):
        row = {
            "chosen": [{"role": "user", "content": "p"}, {"role": "assistant", "content": "c"}],
            "rejected": [{"role": "user", "content": "p"}, {"role": "assistant", "content": "r"}],
        }
        self.assertEqual(validate_preference_pair(row), ("p", "c", "r"))
        self.assertEqual(len(stable_text_hash("p")), 64)
        row["rejected"][0]["content"] = "other"
        with self.assertRaisesRegex(ValueError, "prompts differ"):
            validate_preference_pair(row)

    def test_tokenizer_boundary_and_no_truncation(self):
        messages = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "ok"}]
        tokenizer = FakeTokenizer()
        encoded = openrlhf_encode_response(tokenizer, messages, max_length=100)
        self.assertEqual(encoded.prompt_length, len("USER:x\nASSISTANT:"))
        self.assertEqual(encoded.response_length, len(encoded.input_ids) - encoded.prompt_length)
        self.assertEqual(encoded.input_ids[-1], tokenizer.eos_token_id)
        self.assertFalse(tokenizer.last_kwargs["truncation"])
        with self.assertRaisesRegex(ValueError, "untruncated sequence length"):
            openrlhf_encode_response(tokenizer, messages, max_length=22)

    def test_vllm_response_token_extraction(self):
        class Candidate:
            def __init__(self, value):
                self.logprob = value

        encoded = EncodedResponse([10, 11, 12, 13], 2, 2, "")
        output = type(
            "Output",
            (),
            {
                "prompt_logprobs": [
                    None,
                    {11: Candidate(-9)},
                    {12: Candidate(-1.5)},
                    {13: Candidate(-2)},
                ]
            },
        )()
        self.assertEqual(extract_vllm_response_logprob(output, encoded), -3.5)
        output.prompt_logprobs[3] = {99: Candidate(-2)}
        with self.assertRaisesRegex(ValueError, "omitted actual token"):
            extract_vllm_response_logprob(output, encoded)


if __name__ == "__main__":
    unittest.main()
