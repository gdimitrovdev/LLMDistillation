# Java Test Assertion Generation with Knowledge Distillation

This repository contains tools for fine-tuning a CodeT5 model to generate test assertions, and for implementing knowledge distillation to create a smaller, more efficient model.

## Project Overview

The goal of this project is to automatically generate assertions for Java unit tests. We use a two-stage approach:

1. Fine-tune a pre-trained CodeT5 model (teacher) on test assertion generation
2. Use knowledge distillation to transfer this capability to a smaller model (student)

## Repository Structure

```
.
├── dataset_preparation.py          # Prepares the dataset from raw Java repositories
├── train_teacher_model.py          # Fine-tunes the teacher model
├── codet5_prediction_with_assertions.py  # Generates teacher predictions and compresses logits
└── data/                           # Directory for datasets
```

The `knowledge_distillation.py` file is not provided - implementing this is the main task for students.

## Setup

### Requirements

```
torch>=1.13.1
transformers>=4.20.0
numpy>=1.20.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.62.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Workflow

### 1. Dataset Preparation

The dataset preparation script extracts test methods, focal code, and assertions from Java repositories:

```bash
python dataset_preparation.py \
  --input_path "data/prepared-repositories.jsonl" \
  --output_path "data/assertion_dataset.jsonl" \
  --show_example
```

### 2. Fine-tuning the Teacher Model

The teacher model is a pre-trained CodeT5 model fine-tuned on the assertion generation task:

```bash
python train_teacher_model.py \
  --data_path "data/assertion_dataset.jsonl" \
  --output_dir "./teacher_model" \
  --model_name "Salesforce/codet5-small" \
  --epochs 1 \
  --batch_size 16 \
  --fp16
```

Key parameters:
- `--data_path`: Path to the prepared dataset
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--fp16`: Use mixed precision training (faster)
- `--track_batch_metrics`: Show per-batch metrics

### 3. Generating Teacher Predictions

After fine-tuning, generate predictions and compress logits for knowledge distillation:

```bash
python codet5_prediction_with_assertions.py \
  --data_path "data/assertion_dataset.jsonl" \
  --output_dir "./teacher_output" \
  --model_name "./teacher_model/best_model" \
  --batch_size 16 \
  --fp16 \
  --precision_bits 4
```

Key parameters:
- `--precision_bits`: Compression level (2=highest compression, 8=best quality)
- `--max_samples`: Optional limit on dataset size (e.g., `--max_samples 10000`)

### 4. Knowledge Distillation (Student Task)

Students must implement the knowledge distillation process in `knowledge_distillation.py`. The implementation should:

1. Load the dataset with teacher predictions and compressed logits
2. Create a smaller student model
3. Train the student to mimic the teacher's outputs
4. Evaluate the student model's performance

## Data Format

### Input Dataset (`assertion_dataset.jsonl`)

Each line contains a JSON object with:
- `focal_file`: The code under test
- `test_method_masked`: The test method without assertions
- `assertions`: Array of reference assertions
- `method_under_test`: Identified method being tested

### Teacher Predictions (`dataset_with_predictions.jsonl`)

Each line contains the original data plus:
- `teacher_prediction`: Raw assertion text from the teacher model
- `teacher_parsed_assertions`: Parsed individual assertions
- `teacher_metrics`: Evaluation metrics comparing to ground truth
- `teacher_logits`: Compressed logits for knowledge distillation

## Output Files

### Teacher Model Training
- `teacher_model/best_model/`: Best checkpoint based on validation loss
- `teacher_model/training_metrics.csv`: Detailed training metrics
- `teacher_model/loss_curves.png`: Training and validation loss curves

### Teacher Prediction Generation
- `teacher_output/dataset_with_predictions.jsonl`: Dataset with predictions and logits
- `teacher_output/prediction_metrics.json`: Overall metrics
- `teacher_output/teacher_predictions.csv`: Per-example metrics in CSV format
- `teacher_output/visualizations/`: Charts of performance metrics

## Tensor Compression for Knowledge Distillation

### Compression Techniques

The `codet5_prediction_with_assertions.py` script uses advanced compression techniques to reduce the size of teacher logits:

1. **Quantization**: Converting 32-bit floating-point values to lower precision
   - 8-bit (256 discrete values): Good balance of quality and compression
   - 4-bit (16 discrete values): Higher compression, slight quality loss
   - 2-bit (4 discrete values): Maximum compression, significant quality loss

2. **Sparsity Exploitation**:
   - Values below a threshold are set to zero
   - When >90% of values are zero, uses sparse representation
   - Stores only non-zero values and their indices

3. **Value Range Normalization**:
   - Maps all values to [0,1] range before quantization
   - Stores original min/max for decompression

4. **Bit Packing**:
   - Packs multiple low-precision values into single bytes
   - 4-bit: Two values per byte
   - 2-bit: Four values per byte

5. **Zlib Compression**:
   - Applied to the quantized binary data
   - Uses maximum compression level (9)

6. **Base64 Encoding**:
   - Final binary data is encoded as ASCII text
   - Enables storage in JSON format

### Decompression Implementation

Students must implement a decompression function that reverses these steps. Here's a pseudocode outline:

```python
def decompress_tensor(compressed_data):
    # 1. Get format and metadata
    format_type = compressed_data["format"]  # "sparse" or "quantized_Nbit"
    shape = compressed_data["shape"]
    
    # 2. Decode from base64
    binary_data = base64_decode(compressed_data["data"])
    
    # 3. Decompress with zlib
    decompressed_bytes = zlib_decompress(binary_data)
    
    # 4. Handle different formats
    if format_type == "sparse":
        # 4a. For sparse format
        sparse_data = json_decode(decompressed_bytes)
        indices = sparse_data["indices"]
        values = sparse_data["values"]
        
        # Create tensor with zeros
        tensor = create_zeros_tensor(shape)
        
        # Fill non-zero values
        for idx, val in zip(indices, values):
            tensor[idx] = val
            
    elif "quantized" in format_type:
        # 4b. For quantized format
        min_val = compressed_data["min_val"]
        max_val = compressed_data["max_val"]
        bits = extract_bits_from_format(format_type)  # 2, 4, or 8
        
        if bits == 8:
            # Direct conversion for 8-bit
            quantized = np_frombuffer(decompressed_bytes, dtype=uint8)
            normalized = quantized / 255.0
            
        elif bits == 4:
            # Unpack 4-bit values (2 per byte)
            packed = np_frombuffer(decompressed_bytes, dtype=uint8)
            quantized = unpack_4bit_values(packed, shape)
            normalized = quantized / 15.0
            
        elif bits == 2:
            # Unpack 2-bit values (4 per byte)
            packed = np_frombuffer(decompressed_bytes, dtype=uint8)
            quantized = unpack_2bit_values(packed, shape)
            normalized = quantized / 3.0
        
        # Denormalize to original range
        tensor = normalized * (max_val - min_val) + min_val
        tensor = reshape_tensor(tensor, shape)
    
    return tensor
```

A test script (`decompression_test.py`) is automatically generated to help verify your implementation.

### Compression Performance

The compression ratio depends on the precision bits and data characteristics:
- 2-bit precision: ~15-25x compression
- 4-bit precision: ~8-12x compression
- 8-bit precision: ~4-6x compression

For knowledge distillation, 4-bit precision is typically sufficient as the relative rankings of logits values are preserved, which is what matters for distillation loss.

## Knowledge Distillation Implementation Notes

Students should implement:

1. A smaller T5-based model architecture
2. Loading and decompressing the teacher logits
3. Distillation loss function (combination of hard and soft targets)
4. Training loop with proper evaluation
5. Comparison of teacher vs. student performance

The distillation process should use both:
- Hard targets (cross-entropy with ground truth)
- Soft targets (KL divergence between teacher and student probabilities)

## Evaluation

The student implementation will be evaluated based on:
1. Compression ratio (student model size / teacher model size)
2. Performance gap (student accuracy / teacher accuracy)
3. Inference speed improvement
4. Quality of implementation and documentation