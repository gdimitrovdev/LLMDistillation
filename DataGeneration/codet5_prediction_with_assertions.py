#!/usr/bin/env python
"""
Script to generate predictions from CodeT5 with optimized tensor compression.
"""

import argparse
import json
import os
import re
import torch
import numpy as np
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import zlib
import io


class AssertionDataset(Dataset):
    """Dataset for assertion generation task"""

    def __init__(self, data, tokenizer, max_src_length=1024, max_tgt_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Construct input: We combine focal code, test method without assertions
        input_text = f"FOCAL CODE:\n{item['focal_file']}\n\nTEST METHOD:\n{item['test_method_masked']}"

        # Target: The assertions that need to be generated
        target_text = "\n".join(item['assertions'])

        # Tokenize inputs
        source_encoding = self.tokenizer(
            input_text,
            max_length=self.max_src_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize targets
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_tgt_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = source_encoding["input_ids"].squeeze()
        attention_mask = source_encoding["attention_mask"].squeeze()
        labels = target_encoding["input_ids"].squeeze()

        # Replace padding token id with -100 so it's ignored in loss computation
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "original_input": input_text,
            "original_target": target_text,
            "idx": idx,
        }


def load_dataset(jsonl_path):
    """Load data from JSONL file"""
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def compress_tensor_optimized(tensor, precision_bits=8, sparsity_threshold=0.01):
    """
    Highly optimized tensor compression using quantization, sparsity, and better compression

    Args:
        tensor: The input tensor to compress
        precision_bits: Number of bits for quantization (8, 4, or 2)
        sparsity_threshold: Values below this threshold are treated as zeros

    Returns:
        Dictionary with compressed tensor data and metadata
    """
    # Convert to numpy
    array = tensor.cpu().detach().numpy()

    # Get the shape and original dtype
    original_shape = array.shape
    original_dtype = str(array.dtype)

    # Find min and max for normalization
    min_val = float(np.min(array))
    max_val = float(np.max(array))

    # Apply sparsity: Set values below threshold to zero
    if sparsity_threshold > 0:
        mask = np.abs(array) < sparsity_threshold
        array[mask] = 0

        # Count non-zero elements
        nnz = np.count_nonzero(array)
        sparsity = 1.0 - (nnz / array.size)

        # If the array is very sparse (>90%), use sparse representation
        if sparsity > 0.9:
            # Find indices and values of non-zero elements
            indices = np.nonzero(array)
            values = array[indices]

            # Pack sparse representation
            sparse_data = {
                'indices': [idx.tolist() for idx in indices],
                'values': values.tolist(),
                'shape': list(original_shape),
                'min': min_val,
                'max': max_val,
                'sparse': True
            }

            # Compress with high compression level
            compressed_json = json.dumps(sparse_data).encode()
            compressed = zlib.compress(compressed_json, level=9)

            return {
                'data': base64.b64encode(compressed).decode('ascii'),
                'format': 'sparse',
                'shape': list(original_shape),
                'original_dtype': original_dtype,
                'sparsity': sparsity
            }

    # Normalize to [0, 1] range
    value_range = max_val - min_val
    if value_range > 0:
        normalized = (array - min_val) / value_range
    else:
        normalized = np.zeros_like(array)

    # Quantize based on precision_bits
    if precision_bits == 8:
        quantized = np.round(normalized * 255).astype(np.uint8)
    elif precision_bits == 4:
        # Pack two 4-bit values into one byte
        quantized = np.round(normalized * 15).astype(np.uint8)
        even_indices = np.arange(0, quantized.size, 2)
        if quantized.size % 2 == 1:
            # Pad with zero if odd number of elements
            quantized = np.append(quantized, 0)
            even_indices = np.arange(0, quantized.size - 1, 2)

        odd_indices = even_indices + 1
        packed = (quantized.flat[even_indices] << 4) | quantized.flat[odd_indices]
        quantized = packed.reshape(-1)
    elif precision_bits == 2:
        # Pack four 2-bit values into one byte
        quantized = np.round(normalized * 3).astype(np.uint8)
        size = quantized.size
        if size % 4 != 0:
            # Pad to multiple of 4
            padding = 4 - (size % 4)
            quantized = np.append(quantized, np.zeros(padding, dtype=np.uint8))

        quantized = quantized.reshape(-1, 4)
        packed = (quantized[:, 0] << 6) | (quantized[:, 1] << 4) | (quantized[:, 2] << 2) | quantized[:, 3]
        quantized = packed

    # Convert to bytes
    byte_array = quantized.tobytes()

    # Apply zlib compression with high compression level
    compressed_bytes = zlib.compress(byte_array, level=9)

    return {
        'data': base64.b64encode(compressed_bytes).decode('ascii'),
        'format': f'quantized_{precision_bits}bit',
        'shape': list(original_shape),
        'original_dtype': original_dtype,
        'min_val': min_val,
        'max_val': max_val,
        'compressed_size': len(compressed_bytes),
        'original_size': array.nbytes
    }


def decompress_tensor_optimized(compressed_data):
    """
    Decompress a tensor that was compressed with the optimized compression

    Args:
        compressed_data: The dictionary with compressed tensor data

    Returns:
        The decompressed PyTorch tensor
    """
    # Handle sparse format
    if compressed_data.get('format') == 'sparse':
        # Decode and decompress
        compressed_bytes = base64.b64decode(compressed_data['data'])
        decompressed_json = zlib.decompress(compressed_bytes)
        sparse_data = json.loads(decompressed_json)

        # Reconstruct sparse tensor
        shape = sparse_data['shape']
        indices = sparse_data['indices']
        values = sparse_data['values']
        min_val = sparse_data['min']
        max_val = sparse_data['max']

        # Create empty array
        array = np.zeros(shape, dtype=np.float32)

        # Fill non-zero values
        for idx, val in zip(zip(*indices), values):
            array[idx] = val

        return torch.tensor(array)

    # Handle quantized format
    if 'quantized' in compressed_data.get('format', ''):
        # Extract parameters
        shape = compressed_data['shape']
        min_val = compressed_data['min_val']
        max_val = compressed_data['max_val']
        precision_bits = int(compressed_data['format'].split('_')[1].replace('bit', ''))

        # Decode and decompress
        compressed_bytes = base64.b64decode(compressed_data['data'])
        decompressed_bytes = zlib.decompress(compressed_bytes)

        # Convert to numpy array
        if precision_bits == 8:
            # Direct 8-bit quantization
            quantized = np.frombuffer(decompressed_bytes, dtype=np.uint8)
            normalized = quantized.astype(np.float32) / 255.0

        elif precision_bits == 4:
            # Unpack 4-bit values
            packed = np.frombuffer(decompressed_bytes, dtype=np.uint8)
            total_values = np.prod(shape)

            # Create array for unpacked values
            quantized = np.zeros(total_values, dtype=np.uint8)

            # Unpack values
            even_indices = np.arange(0, total_values, 2)
            odd_indices = np.minimum(even_indices + 1, total_values - 1)

            # Extract 4-bit values
            quantized[even_indices] = (packed >> 4) & 0xF
            if odd_indices[-1] < total_values:
                quantized[odd_indices] = packed & 0xF

            normalized = quantized.astype(np.float32) / 15.0

        elif precision_bits == 2:
            # Unpack 2-bit values
            packed = np.frombuffer(decompressed_bytes, dtype=np.uint8)
            total_values = np.prod(shape)

            # Create array for unpacked values
            quantized = np.zeros(total_values, dtype=np.uint8)

            # Calculate number of complete bytes
            num_complete_bytes = total_values // 4

            # Unpack each byte into 4 values
            for i in range(num_complete_bytes):
                byte = packed[i]
                base_idx = i * 4
                quantized[base_idx] = (byte >> 6) & 0x3
                quantized[base_idx + 1] = (byte >> 4) & 0x3
                quantized[base_idx + 2] = (byte >> 2) & 0x3
                quantized[base_idx + 3] = byte & 0x3

            # Handle remaining values
            remaining = total_values % 4
            if remaining > 0:
                byte = packed[num_complete_bytes]
                base_idx = num_complete_bytes * 4
                for j in range(remaining):
                    shift = 6 - j * 2
                    quantized[base_idx + j] = (byte >> shift) & 0x3

            normalized = quantized.astype(np.float32) / 3.0

        # Denormalize
        array = normalized * (max_val - min_val) + min_val

        # Reshape to original shape
        array = array.reshape(shape)

        # Convert to tensor
        return torch.tensor(array, dtype=torch.float32)

    # Fallback to original method
    binary_data = base64.b64decode(compressed_data['data'])
    stream = io.BytesIO(binary_data)
    loaded = np.load(stream, allow_pickle=True)
    array = loaded['data']

    # Restore the original dtype
    if compressed_data.get('dtype') == 'float16':
        array = array.astype(np.float16)
    elif compressed_data.get('dtype') == 'float32':
        array = array.astype(np.float32)

    # Convert to tensor
    tensor = torch.tensor(array)

    return tensor


def normalize_assertion(assertion):
    """Normalize assertion text for more reliable comparison"""
    # Remove whitespace
    assertion = re.sub(r'\s+', ' ', assertion).strip()

    # Remove variable names in certain cases
    assertion = re.sub(r'assertEquals\(\s*[^,]+,\s*([^)]+)\)', r'assertEquals(VALUE, \1)', assertion)

    # Normalize assertion method names
    assertion = re.sub(r'assert(Equals|That|True|False)', r'assert\1', assertion, flags=re.IGNORECASE)

    return assertion


def calculate_similarity(reference, candidate):
    """Calculate string similarity using SequenceMatcher"""
    return SequenceMatcher(None, reference, candidate).ratio()


def classify_assertion_type(assertion):
    """Classify the type of assertion"""
    assertion = assertion.lower()

    if "assertEquals" in assertion or "assertThat" in assertion and ".isEqualTo" in assertion:
        return "equality"
    elif "assertTrue" in assertion:
        return "truth"
    elif "assertFalse" in assertion:
        return "falsity"
    elif "assertNull" in assertion:
        return "null"
    elif "assertNotNull" in assertion:
        return "not_null"
    elif "assertThrows" in assertion:
        return "exception"
    elif "assertSame" in assertion:
        return "same"
    elif "assertNotSame" in assertion:
        return "not_same"
    elif "assertArrayEquals" in assertion:
        return "array_equality"
    else:
        return "other"


def evaluate_assertions(generated_assertions, reference_assertions):
    """Evaluate the quality of generated assertions against reference assertions"""

    # Parse individual assertions if provided as multiline string
    if isinstance(generated_assertions, str):
        # Split by semicolons or newlines
        generated_list = re.split(r';|\n', generated_assertions)
        generated_list = [a.strip() + ';' for a in generated_list if a.strip()]
    else:
        generated_list = generated_assertions

    if isinstance(reference_assertions, str):
        reference_list = re.split(r';|\n', reference_assertions)
        reference_list = [a.strip() + ';' for a in reference_list if a.strip()]
    else:
        reference_list = reference_assertions

    # Normalize assertions
    normalized_generated = [normalize_assertion(a) for a in generated_list]
    normalized_reference = [normalize_assertion(a) for a in reference_list]

    # Calculate exact matches (accuracy)
    exact_matches = 0
    for gen in normalized_generated:
        if gen in normalized_reference:
            exact_matches += 1

    # Calculate similarity scores
    similarity_scores = []
    for gen in normalized_generated:
        best_sim = 0
        for ref in normalized_reference:
            sim = calculate_similarity(gen, ref)
            best_sim = max(best_sim, sim)
        similarity_scores.append(best_sim)

    # Classify assertion types
    gen_types = [classify_assertion_type(a) for a in generated_list]
    ref_types = [classify_assertion_type(a) for a in reference_list]

    # Count assertion types
    gen_type_counts = {t: gen_types.count(t) for t in set(gen_types)}
    ref_type_counts = {t: ref_types.count(t) for t in set(ref_types)}

    # Calculate metrics
    precision = exact_matches / len(normalized_generated) if normalized_generated else 0
    recall = exact_matches / len(normalized_reference) if normalized_reference else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = exact_matches / max(len(normalized_generated), len(normalized_reference)) if max(
        len(normalized_generated), len(normalized_reference)) > 0 else 0

    return {
        "exact_matches": exact_matches,
        "generated_count": len(normalized_generated),
        "reference_count": len(normalized_reference),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "similarity_score_avg": sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0,
        "generated_type_counts": gen_type_counts,
        "reference_type_counts": ref_type_counts,
        "similarity_scores": similarity_scores
    }


def generate_and_save_predictions(model, tokenizer, dataloader, output_file, device, precision_bits=4,
                                  sparsity_threshold=0.01):
    """Generate predictions and save them with compressed logits inline in the JSONL file"""

    print(f"Generating predictions with optimized compression (using {precision_bits}-bit precision)...")
    print(f"Output will be saved to: {output_file}")

    model.to(device)
    model.eval()

    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Process in batches
    all_entries = []
    all_metrics = {
        "exact_matches": 0,
        "generated_count": 0,
        "reference_count": 0,
        "similarity_scores": [],
        "accuracy_scores": [],
        "f1_scores": []
    }

    # Track the size of raw vs compressed data
    total_raw_size = 0
    total_compressed_size = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating predictions")):
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            indices = batch["idx"]

            # Get logits (raw model outputs)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            teacher_logits = outputs.logits

            # Generate actual predictions
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )

            # Process each item in the batch
            for i in range(len(input_ids)):
                idx = indices[i].item()
                original_entry = dataloader.dataset.data[idx]

                # Get individual logits tensor
                item_logits = teacher_logits[i]

                # Calculate raw size
                raw_size = item_logits.element_size() * item_logits.nelement()
                total_raw_size += raw_size

                # Compress logits with optimized compression
                compressed_logits = compress_tensor_optimized(
                    item_logits,
                    precision_bits=precision_bits,
                    sparsity_threshold=sparsity_threshold
                )

                # Calculate compressed size
                compressed_size = len(json.dumps(compressed_logits))
                total_compressed_size += compressed_size

                # Decode the generated text
                generated_text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)

                # Evaluate the prediction
                reference_assertions = original_entry['assertions']

                try:
                    metrics = evaluate_assertions(generated_text, reference_assertions)
                except Exception as e:
                    print(f"Error evaluating assertions for item {idx}: {str(e)}")
                    metrics = {
                        "exact_matches": 0,
                        "generated_count": 0,
                        "reference_count": 0,
                        "precision": 0,
                        "recall": 0,
                        "f1": 0,
                        "accuracy": 0,
                        "similarity_score_avg": 0,
                        "generated_type_counts": {},
                        "reference_type_counts": {},
                        "similarity_scores": []
                    }

                # Update overall metrics
                all_metrics["exact_matches"] += metrics["exact_matches"]
                all_metrics["generated_count"] += metrics["generated_count"]
                all_metrics["reference_count"] += metrics["reference_count"]
                all_metrics["similarity_scores"].extend(metrics["similarity_scores"])
                all_metrics["accuracy_scores"].append(metrics["accuracy"])
                all_metrics["f1_scores"].append(metrics["f1"])

                # Create an entry with original data + predictions + metrics + compressed logits
                entry = original_entry.copy()
                entry["model_prediction"] = generated_text
                entry["prediction_metrics"] = {
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "accuracy": metrics["accuracy"],
                    "similarity": metrics["similarity_score_avg"],
                    "exact_matches": metrics["exact_matches"]
                }
                entry["compressed_logits"] = compressed_logits

                # Add to the collection
                all_entries.append(entry)

            # Periodically save progress
            if (batch_idx + 1) % 20 == 0 or batch_idx == len(dataloader) - 1:
                # Save all entries
                with open(output_file, 'w') as f:
                    for entry in all_entries:
                        f.write(json.dumps(entry) + '\n')

                # Calculate overall metrics
                if all_metrics["generated_count"] > 0 and all_metrics["reference_count"] > 0:
                    avg_precision = all_metrics["exact_matches"] / all_metrics["generated_count"]
                    avg_recall = all_metrics["exact_matches"] / all_metrics["reference_count"]
                    avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (
                                                                                                          avg_precision + avg_recall) > 0 else 0
                    total_count = all_metrics["generated_count"] + all_metrics["reference_count"]
                    overall_accuracy = all_metrics["exact_matches"] * 2 / total_count if total_count > 0 else 0
                else:
                    avg_precision = 0
                    avg_recall = 0
                    avg_f1 = 0
                    overall_accuracy = 0

                avg_similarity = sum(all_metrics["similarity_scores"]) / len(all_metrics["similarity_scores"]) if \
                all_metrics["similarity_scores"] else 0
                avg_per_sample_accuracy = sum(all_metrics["accuracy_scores"]) / len(all_metrics["accuracy_scores"]) if \
                all_metrics["accuracy_scores"] else 0

                # Progress stats
                compression_ratio = total_raw_size / total_compressed_size if total_compressed_size > 0 else 0

                print(f"[Batch {batch_idx + 1}/{len(dataloader)}] Stats so far:")
                print(f"  Accuracy: {avg_per_sample_accuracy:.4f}")
                print(f"  Precision: {avg_precision:.4f}")
                print(f"  Recall: {avg_recall:.4f}")
                print(f"  F1: {avg_f1:.4f}")
                print(f"  Avg similarity: {avg_similarity:.4f}")
                print(
                    f"  Compression ratio: {compression_ratio:.2f}x (Raw: {total_raw_size / 1024 / 1024:.2f}MB, Compressed: {total_compressed_size / 1024 / 1024:.2f}MB)")

    # Calculate final metrics
    if all_metrics["generated_count"] > 0 and all_metrics["reference_count"] > 0:
        overall_precision = all_metrics["exact_matches"] / all_metrics["generated_count"]
        overall_recall = all_metrics["exact_matches"] / all_metrics["reference_count"]
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (
                                                                                                                  overall_precision + overall_recall) > 0 else 0
        total_count = all_metrics["generated_count"] + all_metrics["reference_count"]
        overall_accuracy = all_metrics["exact_matches"] * 2 / total_count if total_count > 0 else 0
    else:
        overall_precision = 0
        overall_recall = 0
        overall_f1 = 0
        overall_accuracy = 0

    avg_similarity = sum(all_metrics["similarity_scores"]) / len(all_metrics["similarity_scores"]) if all_metrics[
        "similarity_scores"] else 0
    avg_per_sample_accuracy = sum(all_metrics["accuracy_scores"]) / len(all_metrics["accuracy_scores"]) if all_metrics[
        "accuracy_scores"] else 0
    avg_f1 = sum(all_metrics["f1_scores"]) / len(all_metrics["f1_scores"]) if all_metrics["f1_scores"] else 0

    final_metrics = {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "accuracy": overall_accuracy,
        "avg_per_sample_accuracy": avg_per_sample_accuracy,
        "avg_similarity": avg_similarity,
        "avg_per_sample_f1": avg_f1,
        "total_exact_matches": all_metrics["exact_matches"],
        "total_generated": all_metrics["generated_count"],
        "total_reference": all_metrics["reference_count"],
        "compression_stats": {
            "raw_size_mb": total_raw_size / 1024 / 1024,
            "compressed_size_mb": total_compressed_size / 1024 / 1024,
            "compression_ratio": total_raw_size / total_compressed_size if total_compressed_size > 0 else 0,
            "precision_bits": precision_bits
        }
    }

    # Save final metrics
    metrics_file = os.path.join(os.path.dirname(output_file), "prediction_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(final_metrics, f, indent=2)

    # Generate visualizations
    try:
        visualize_results(all_metrics, final_metrics, os.path.dirname(output_file))
    except Exception as e:
        print(f"Warning: Could not generate visualizations: {str(e)}")

    # Create a decompression test script
    test_file = os.path.join(os.path.dirname(output_file), "decompression_test.py")
    with open(test_file, 'w') as f:
        f.write("""
#!/usr/bin/env python
# Test script to verify logits decompression works
import json
import torch
import numpy as np
import base64
import zlib
import io
import os

def decompress_tensor_optimized(compressed_data):
    # Handle sparse format
    if compressed_data.get('format') == 'sparse':
        # Decode and decompress
        compressed_bytes = base64.b64decode(compressed_data['data'])
        decompressed_json = zlib.decompress(compressed_bytes)
        sparse_data = json.loads(decompressed_json)

        # Reconstruct sparse tensor
        shape = sparse_data['shape']
        indices = sparse_data['indices']
        values = sparse_data['values']
        min_val = sparse_data['min']
        max_val = sparse_data['max']

        # Create empty array
        array = np.zeros(shape, dtype=np.float32)

        # Fill non-zero values
        for idx, val in zip(zip(*indices), values):
            array[idx] = val

        return torch.tensor(array)

    # Handle quantized format
    if 'quantized' in compressed_data.get('format', ''):
        # Extract parameters
        shape = compressed_data['shape']
        min_val = compressed_data['min_val']
        max_val = compressed_data['max_val']
        precision_bits = int(compressed_data['format'].split('_')[1].replace('bit', ''))

        # Decode and decompress
        compressed_bytes = base64.b64decode(compressed_data['data'])
        decompressed_bytes = zlib.decompress(compressed_bytes)

        # Convert to numpy array
        if precision_bits == 8:
            # Direct 8-bit quantization
            quantized = np.frombuffer(decompressed_bytes, dtype=np.uint8)
            normalized = quantized.astype(np.float32) / 255.0

        elif precision_bits == 4:
            # Unpack 4-bit values
            packed = np.frombuffer(decompressed_bytes, dtype=np.uint8)
            total_values = np.prod(shape)

            # Create array for unpacked values
            quantized = np.zeros(total_values, dtype=np.uint8)

            # Unpack values
            even_indices = np.arange(0, total_values, 2)
            odd_indices = np.minimum(even_indices + 1, total_values - 1)

            # Extract 4-bit values
            quantized[even_indices] = (packed >> 4) & 0xF
            if odd_indices[-1] < total_values:
                quantized[odd_indices] = packed & 0xF

            normalized = quantized.astype(np.float32) / 15.0

        elif precision_bits == 2:
            # Unpack 2-bit values
            packed = np.frombuffer(decompressed_bytes, dtype=np.uint8)
            total_values = np.prod(shape)

            # Create array for unpacked values
            quantized = np.zeros(total_values, dtype=np.uint8)

            # Calculate number of complete bytes
            num_complete_bytes = total_values // 4

            # Unpack each byte into 4 values
            for i in range(num_complete_bytes):
                byte = packed[i]
                base_idx = i * 4
                quantized[base_idx] = (byte >> 6) & 0x3
                quantized[base_idx + 1] = (byte >> 4) & 0x3
                quantized[base_idx + 2] = (byte >> 2) & 0x3
                quantized[base_idx + 3] = byte & 0x3

            # Handle remaining values
            remaining = total_values % 4
            if remaining > 0:
                byte = packed[num_complete_bytes]
                base_idx = num_complete_bytes * 4
                for j in range(remaining):
                    shift = 6 - j * 2
                    quantized[base_idx + j] = (byte >> shift) & 0x3

            normalized = quantized.astype(np.float32) / 3.0

        # Denormalize
        array = normalized * (max_val - min_val) + min_val

        # Reshape to original shape
        array = array.reshape(shape)

        # Convert to tensor
        return torch.tensor(array, dtype=torch.float32)

    # Fallback to original method for other formats
    if 'data' in compressed_data:
        try:
            binary_data = base64.b64decode(compressed_data['data'])
            stream = io.BytesIO(binary_data)
            loaded = np.load(stream, allow_pickle=True)
            array = loaded['data']

            # Restore the original dtype
            if compressed_data.get('dtype') == 'float16':
                array = array.astype(np.float16)
            elif compressed_data.get('dtype') == 'float32':
                array = array.astype(np.float32)

            # Convert to tensor
            return torch.tensor(array)
        except Exception as e:
            print(f"Error decompressing data: {e}")
            return None

    return None

print("Testing decompression of logits...")

# Find the dataset file
dataset_file = "dataset_with_predictions.jsonl"
if not os.path.exists(dataset_file):
    dataset_file = os.path.join(os.path.dirname(__file__), dataset_file)

# Load the first entry
with open(dataset_file, 'r') as f:
    first_entry = json.loads(f.readline())

print("Entry loaded successfully.")

# Decompress the logits
try:
    logits = decompress_tensor_optimized(first_entry["compressed_logits"])

    # Print info
    print(f"Successfully decompressed logits!")
    print(f"Shape: {logits.shape}")
    print(f"Data type: {logits.dtype}")
    print(f"Min value: {logits.min().item():.4f}")
    print(f"Max value: {logits.max().item():.4f}")
    print(f"Sample values (first 5): {logits.flatten()[:5].tolist()}")

    compression_format = first_entry["compressed_logits"].get("format", "unknown")
    print(f"Compression format: {compression_format}")
    if 'compressed_size' in first_entry["compressed_logits"] and 'original_size' in first_entry["compressed_logits"]:
        original_size = first_entry["compressed_logits"]["original_size"]
        compressed_size = first_entry["compressed_logits"]["compressed_size"]
        ratio = original_size / compressed_size if compressed_size > 0 else 0
        print(f"Compression ratio: {ratio:.2f}x")
        print(f"Original size: {original_size/1024/1024:.2f} MB")
        print(f"Compressed size: {compressed_size/1024/1024:.2f} MB")
except Exception as e:
    print(f"Error decompressing logits: {e}")
""")

    # Make the test script executable
    os.chmod(test_file, 0o755)

    print("\nCreated decompression test script. Run it with:")
    print(f"python {test_file}")

    return all_entries, final_metrics


def visualize_results(all_metrics, final_metrics, output_dir):
    """Create visualizations of the evaluation results"""

    # Create visualizations directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # 1. Overall metrics bar chart
    plt.figure(figsize=(10, 6))
    metrics = ["precision", "recall", "f1", "accuracy", "avg_similarity"]
    values = [final_metrics[m] for m in metrics]

    sns.barplot(x=metrics, y=values)
    plt.title("Overall Assertion Generation Performance")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(vis_dir, "overall_metrics.png"))
    plt.close()

    # 2. Similarity score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(all_metrics["similarity_scores"], bins=20, kde=True)
    plt.title("Distribution of Similarity Scores")
    plt.xlabel("Similarity Score")
    plt.ylabel("Count")
    plt.savefig(os.path.join(vis_dir, "similarity_distribution.png"))
    plt.close()

    # 3. Accuracy score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(all_metrics["accuracy_scores"], bins=20, kde=True)
    plt.title("Distribution of Accuracy Scores")
    plt.xlabel("Accuracy Score")
    plt.ylabel("Count")
    plt.savefig(os.path.join(vis_dir, "accuracy_distribution.png"))
    plt.close()

    # 4. F1 score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(all_metrics["f1_scores"], bins=20, kde=True)
    plt.title("Distribution of F1 Scores")
    plt.xlabel("F1 Score")
    plt.ylabel("Count")
    plt.savefig(os.path.join(vis_dir, "f1_distribution.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate predictions with optimized compression")

    # Data and model args
    parser.add_argument("--data_path", type=str, required=True, help="Path to assertion dataset jsonl")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save results")
    parser.add_argument("--model_name", type=str, default="Salesforce/codet5-base", help="CodeT5 model to use")

    # Compression args
    parser.add_argument("--precision_bits", type=int, default=4, choices=[2, 4, 8],
                        help="Bits per value for compression (2=highest compression, 8=best quality)")
    parser.add_argument("--sparsity_threshold", type=float, default=0.01,
                        help="Values below this threshold will be treated as zeros")

    # Processing args
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--max_src_length", type=int, default=1024, help="Max source sequence length")
    parser.add_argument("--max_tgt_length", type=int, default=512, help="Max target sequence length")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for generation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision for inference")

    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "dataset_with_predictions.jsonl")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    data = load_dataset(args.data_path)
    print(f"Loaded {len(data)} examples")

    # Create dataset and dataloader
    print(f"Loading tokenizer for {args.model_name}...")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)

    dataset = AssertionDataset(
        data,
        tokenizer,
        max_src_length=args.max_src_length,
        max_tgt_length=args.max_tgt_length
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Keep order for consistent indexing
        num_workers=args.num_workers,
        pin_memory=True  # Speed up data transfer to GPU
    )

    # Load model
    print(f"Loading pre-trained model {args.model_name}...")
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    # Set up mixed precision if requested
    if args.fp16:
        print("Using mixed precision (FP16)")
        model = model.half()

    # Generate predictions and save with metrics
    entries, metrics = generate_and_save_predictions(
        model,
        tokenizer,
        dataloader,
        output_file,
        device,
        args.precision_bits,
        args.sparsity_threshold
    )

    # Print final metrics
    print("\nFinal Metrics:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue:.4f}" if isinstance(subvalue, float) else f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    print(f"\nResults saved to {args.output_dir}")
    print(f"  - Predictions and metrics with compressed logits: {output_file}")
    print(f"  - Overall metrics: {os.path.join(args.output_dir, 'prediction_metrics.json')}")
    print(f"  - Visualizations: {os.path.join(args.output_dir, 'visualizations')}")
    print(f"  - Decompression test script: {os.path.join(args.output_dir, 'decompression_test.py')}")


if __name__ == "__main__":
    main()