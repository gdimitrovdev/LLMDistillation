# Java Test Assertion Generation

This repository contains code and datasets for fine-tuning code models on Java test assertion generation. The models are trained to predict appropriate test assertions based on a focal Java class and a test method without assertions.

## Repository Structure

```
.
├── README.md
├── compress.py                           # Compression/decompression utilities for logits
├── train_codet5_assertions.py            # Training script for CodeT5/CodeT5+
└── data/
    ├── codet5/
        ├── distillation_data_training.jsonl  # Training data with predictions and compressed logits
        └── distillation_data_validation.jsonl # Validation data with predictions and compressed logits
```

The data folder will be initially gitignored, you can look at the README for the main repository to learn how to get that data.

## Dataset Format

### Original Dataset (`assertion_dataset_fixed.jsonl`)

Each entry in the original dataset is a JSON object with the following structure:

```json
{
  "focal_file": "Java code of the class under test",
  "test_method_masked": "Test method with assertions removed",
  "assertions": ["List of assertion statements that were masked out"]
}
```

The dataset was split into approximately 9K examples for training and 1K examples for validation.

### Distillation Datasets

Each model directory contains processed distillation datasets with the following format:

```json
{
  "header": {
    "dataset_type": "training|validation",
    "item_count": 9000,
    "compression": {
      "original_size_mb": 500.0,
      "compressed_size_mb": 10.0,
      "estimated_file_size_mb": 11.0,
      "overall_compression_ratio": 50.0,
      "avg_compression_ratio": 50.0,
      "format_counts": {"lz4": 9000}
    },
    "created_at": "2023-...",
    "args": {
      "bits": 4,
      "model_type": "codet5|codet5plus|codegen",
      "fp16": true
    }
  }
},
{
  "focal_file": "Java code of the class under test",
  "test_method_masked": "Test method with assertions removed",
  "original_target": "Original assertions from the dataset",
  "predicted_assertions": "Model-generated assertions",
  "model_type": "codet5|codet5plus|codegen",
  "compressed_logits": {
    "format": "lz4",
    "compression": {
      "bits": 4,
      "original_size_bytes": 1000000,
      "bit_compressed_size_bytes": 125000,
      "final_size_bytes": 25000,
      "compression_ratio": 40.0
    },
    "shape": [100, 50000],
    "data_encoded": {...},
    "bits": 4
  }
}
```

## Scripts

### `compress.py`

Contains utilities for compressing and decompressing model logits:

- `compress_logits(logits, bits)`: Compresses logits tensor using bit-depth reduction (4, 8, 16, or 32 bits) followed by LZ4 compression
- `decompress_logits(compressed_logits)`: Decompresses logits back to their original form

### `train_codet5_assertions.py`

Script for fine-tuning CodeT5 or CodeT5+ models on Java assertion generation:

```bash
python train_codet5_assertions.py \
  --data_path data/assertion_dataset_fixed.jsonl \
  --output_dir output_codet5 \
  --model_name Salesforce/codet5-base \
  --model_type codet5 \
  --epochs 5 \
  --batch_size 8 \
  --fp16 \
  --compression_bits 4 \
  --create_distillation_dataset
```

### `train_codegen_assertions.py`

Script for fine-tuning CODEGEN on Java assertion generation, adapted for causal language models:

```bash
python train_codegen_assertions.py \
  --data_path data/assertion_dataset_fixed.jsonl \
  --output_dir output_codegen \
  --model_name Salesforce/codegen-350M-mono \
  --model_type codegen \
  --epochs 5 \
  --batch_size 4 \
  --gradient_accumulation_steps 2 \
  --fp16 \
  --compression_bits 4 \
  --create_distillation_dataset
```

## Models

The repository includes fine-tuned models and distillation datasets for:

1. **CodeT5**: Encoder-decoder model pre-trained on code
2. **CodeT5+**: Improved version of CodeT5 with better performance
3. **codegen**: CodeGen is a family of autoregressive language models for program synthesis

All models were fine-tuned on the training split (~9K examples) but not on the validation split (~1K examples). The validation set was used only for evaluation and creating the distillation datasets.

## Compression Details

The model logits are compressed using a 4-bit quantization followed by LZ4 compression, achieving compression ratios around 40-50x without significant quality degradation. This makes the distillation datasets much more manageable in size.

## Usage

### Fine-tuning a Model

```bash
python train_codet5_assertions.py --data_path data/assertion_dataset_fixed.jsonl --output_dir output_model --model_name Salesforce/codet5-base --fp16
```

### Creating Distillation Datasets Only

To create distillation datasets from an already fine-tuned model:

```bash
python train_codet5_assertions.py --data_path data/assertion_dataset_fixed.jsonl --model_path path/to/finetuned/model --output_dir distillation_output --compression_bits 4 --create_distillation_dataset
```
