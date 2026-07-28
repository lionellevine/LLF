# Data card

## Contents

Ten Parquet files contain total chosen/rejected response log-probabilities from one base model and ten public LoRA adapters, plus three exactly derived score columns per adapter. They cover 84,803 preference rows and occupy 47,966,184 bytes. Raw prompts and responses are excluded. SHA-256 hashes preserve prompt and response identity; these identifiers may enable linkage to public source rows and should not be treated as anonymous.

## Source and intended use

Source dataset: `maius/OpenCharacterTraining-data`, revision `2577813a6a435d21051c0548ff2f29dc897212d7`. Open Character Training was released by Sharan Maiya, Henning Bartsch, Nathan Lambert, and Evan Hubinger; see the [paper](https://arxiv.org/abs/2511.01689), [dataset](https://huggingface.co/datasets/maius/OpenCharacterTraining-data), and [code](https://github.com/maiush/OpenCharacterTraining). Intended uses are non-commercial research, reproduction, calibration-method development, and analysis of precomputed likelihood measurements. The bundle is not intended to reconstruct source text or identify contributors.

## Licensing

The MIT license covers code only. The source repository states CC BY-NC-SA/non-commercial research terms unless an underlying source is stricter. This repository does not relicense the measurements or underlying data. Users are responsible for reviewing source provenance and complying with attribution, share-alike, non-commercial, and stricter-source conditions.

## Limitations

The data records one pinned model/adapter/scoring setup. Hashes are linkable identifiers. Log-probabilities may vary across inference engines, attention backends, hardware, and software versions. Calibration is descriptive normalization and does not establish causal mediation or behavioral validity.
