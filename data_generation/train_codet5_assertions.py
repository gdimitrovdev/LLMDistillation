#!/usr/bin/env python
"""
Script to fine-tune CodeT5 models on Java test assertion generation and create distillation dataset.
Ensures test methods are preserved and logits are properly captured.
"""

import argparse
import json
import os
import re
from contextlib import nullcontext

import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    T5ForConditionalGeneration,
    RobertaTokenizer,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from difflib import SequenceMatcher
from compress import compress_logits


class AssertionDataset(Dataset):
    """Dataset for assertion generation that prioritizes preserving test methods"""

    def __init__(self, data, tokenizer, max_src_length=1024, max_tgt_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Extract data fields
        focal_file = item['focal_file']
        test_method_masked = item['test_method_masked']
        assertions = item['assertions']

        # First, tokenize just the test method to determine its token length
        test_method_tokens = self.tokenizer(
            f"TEST METHOD:\n{test_method_masked}",
            add_special_tokens=True,
            truncation=True,  # Add truncation here
            max_length=self.max_src_length,  # Use max_src_length as the limit
            return_tensors="pt"
        )
        test_method_length = test_method_tokens.input_ids.size(1)

        # If test method already exceeds limit (rare but possible), we must truncate it
        if test_method_length >= self.max_src_length - 10:  # Leave room for special tokens
            # Just keep the test method, already truncated
            input_text = f"TEST METHOD:\n{self.tokenizer.decode(test_method_tokens.input_ids[0], skip_special_tokens=True)}"
        else:
            # Determine how much space we have left for the focal file
            space_for_focal = self.max_src_length - test_method_length - 20  # Reserve tokens for prefix and special tokens

            # Format input text based on available space
            if space_for_focal <= 0:
                # Not enough space - use only test method
                input_text = f"TEST METHOD:\n{test_method_masked}"
            else:
                # Tokenize focal file to check its length, with explicit truncation
                focal_tokens = self.tokenizer(
                    focal_file,
                    add_special_tokens=False,
                    truncation=True,  # Add truncation here
                    max_length=space_for_focal,  # Limit to available space
                    return_tensors="pt"
                )

                # Create combined input with truncated focal file if needed
                truncated_focal = self.tokenizer.decode(focal_tokens.input_ids[0], skip_special_tokens=True)
                input_text = f"FOCAL CODE:\n{truncated_focal}\n\nTEST METHOD:\n{test_method_masked}"

        # Target text
        target_text = "\n".join(assertions) if isinstance(assertions, list) else assertions

        # Apply strict truncation at the tokenizer level
        source_encoding = self.tokenizer(
            input_text,
            max_length=self.max_src_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_tgt_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Double-check lengths and force truncation if needed (safety check)
        if source_encoding["input_ids"].size(1) > self.max_src_length:
            source_encoding["input_ids"] = source_encoding["input_ids"][:, :self.max_src_length]
            source_encoding["attention_mask"] = source_encoding["attention_mask"][:, :self.max_src_length]

        if target_encoding["input_ids"].size(1) > self.max_tgt_length:
            target_encoding["input_ids"] = target_encoding["input_ids"][:, :self.max_tgt_length]

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
            "original_target": target_text
        }


def load_dataset(jsonl_path, max_samples=None):
    """Load data from JSONL file with optional sample limit"""
    data = []
    total_lines = 0
    valid_lines = 0

    # First count lines
    with open(jsonl_path, 'r') as f:
        for _ in f:
            total_lines += 1

    # Then load with progress bar
    with open(jsonl_path, 'r') as f:
        for line in tqdm(f, total=total_lines, desc="Loading dataset"):
            if line.strip():
                try:
                    entry = json.loads(line)
                    # Validate required fields are present
                    if 'focal_file' in entry and 'test_method_masked' in entry and 'assertions' in entry:
                        data.append(entry)
                        valid_lines += 1
                        if max_samples and valid_lines >= max_samples:
                            break
                    else:
                        print("Warning: Skipping entry with missing fields")
                except json.JSONDecodeError:
                    print("Warning: Skipping invalid JSON line")

    return data


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

    # Special case handling for empty lists
    if not generated_list or not reference_list:
        return {
            "exact_matches": 0,
            "generated_count": len(generated_list) if generated_list else 0,
            "reference_count": len(reference_list) if reference_list else 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "accuracy": 0,
            "similarity_score_avg": 0,
            "similarity_scores": []
        }

    # Normalize assertions
    normalized_generated = [normalize_assertion(a) for a in generated_list]
    normalized_reference = [normalize_assertion(a) for a in reference_list]

    # Calculate exact matches
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
        "similarity_scores": similarity_scores
    }

def create_distillation_dataset(model, tokenizer, dataloader, device, args, output_path, dataset_type=""):
    """Create a distillation dataset with predictions and logits"""
    model.eval()
    distillation_data = []

    success_count = 0
    failure_count = 0

    # Compression statistics tracking with dynamic format counts
    compression_stats = {
        'format_counts': {},
        'original_size_total': 0,
        'compressed_size_total': 0,
        'compression_ratios': [],
        'sparsity_ratios': []
    }

    # Set up mixed precision if requested
    use_fp16 = args.fp16

    # Import the needed context manager
    from contextlib import nullcontext
    import time

    # Helper function to convert numpy types to Python native types
    def convert_numpy_to_python(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_python(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_to_python(item) for item in obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_numpy_to_python(obj.tolist())
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Creating {dataset_type} distillation dataset")):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Process each example in the batch
            for i in range(input_ids.size(0)):
                try:
                    # Extract original data
                    original_input = batch["original_input"][i]
                    original_target = batch["original_target"][i]

                    # Set up autocast for mixed precision if requested
                    autocast_context = torch.cuda.amp.autocast() if use_fp16 else nullcontext()

                    # Generate prediction with mixed precision
                    with autocast_context:
                        generated_ids = model.generate(
                            input_ids=input_ids[i:i + 1],
                            attention_mask=attention_mask[i:i + 1],
                            max_length=args.max_tgt_length,
                            num_beams=4,
                            early_stopping=True
                        )

                    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                    # Extract logits with the same mixed precision setting
                    logits = extract_logits_from_codet5(
                        model, tokenizer,
                        input_ids[i:i + 1], attention_mask[i:i + 1],
                        original_target, device,
                        max_length=args.max_tgt_length,
                        use_fp16=use_fp16
                    )

                    # Compress logits if successful
                    compressed_logits = None
                    if logits is not None:
                        compressed_logits = compress_logits(
                            logits,
                            args.compression_bits
                        )

                        # Update compression statistics
                        if compressed_logits:
                            format_type = compressed_logits.get('format', 'lz4')  # Default to 'lz4' instead of 'dense'
                            # Initialize the format count if we haven't seen this format before
                            if format_type not in compression_stats['format_counts']:
                                compression_stats['format_counts'][format_type] = 0

                            # Now increment the count
                            compression_stats['format_counts'][format_type] += 1

                            # Get compression info
                            comp_info = compressed_logits.get('compression', {})
                            original_size = comp_info.get('original_size_bytes', 0)
                            final_size = comp_info.get('final_size_bytes', 0)
                            ratio = comp_info.get('compression_ratio', 1.0)

                            compression_stats['original_size_total'] += original_size
                            compression_stats['compressed_size_total'] += final_size
                            compression_stats['compression_ratios'].append(ratio)

                            # Track sparsity for sparse format
                            if format_type == 'sparse':
                                sparsity = comp_info.get('sparsity_ratio', 0.0)
                                compression_stats['sparsity_ratios'].append(sparsity)

                        # Convert any NumPy types to Python native types
                        compressed_logits = convert_numpy_to_python(compressed_logits)
                        success_count += 1
                    else:
                        failure_count += 1

                    # Parse input to extract components
                    focal_file = ""
                    test_method_masked = ""

                    if isinstance(original_input, str):
                        if "FOCAL CODE:" in original_input and "TEST METHOD:" in original_input:
                            parts = original_input.split("FOCAL CODE:")
                            if len(parts) > 1:
                                focal_code_parts = parts[1].split("TEST METHOD:")
                                if len(focal_code_parts) > 1:
                                    focal_file = focal_code_parts[0].strip()
                                    test_method_masked = focal_code_parts[1].strip()
                        elif "TEST METHOD:" in original_input:
                            parts = original_input.split("TEST METHOD:")
                            if len(parts) > 1:
                                test_method_masked = parts[1].strip()

                    # Create distillation item
                    item = {
                        "focal_file": focal_file,
                        "test_method_masked": test_method_masked,
                        "original_target": original_target,
                        "predicted_assertions": generated_text,
                        "model_type": args.model_type,
                        "compressed_logits": compressed_logits
                    }

                    # Add to results
                    distillation_data.append(item)

                    # Print samples occasionally
                    if len(distillation_data) <= 2 or len(distillation_data) % 1000 == 0:
                        print(f"\n{dataset_type.capitalize()} distillation item {len(distillation_data)}:")
                        print(f"Predicted: {generated_text[:100]}..." if len(
                            generated_text) > 100 else f"Predicted: {generated_text}")
                        print(f"Original: {original_target[:100]}..." if len(
                            original_target) > 100 else f"Original: {original_target}")
                        print(f"Has logits: {compressed_logits is not None}")
                        if compressed_logits:
                            format_type = compressed_logits.get('format', 'lz4')
                            bits = compressed_logits.get('bits', 32)
                            comp_info = compressed_logits.get('compression', {})
                            ratio = comp_info.get('compression_ratio', 1.0)
                            print(f"Logits format: {format_type}, {bits}-bit, compression ratio: {ratio:.2f}x")
                            print(
                                f"Original size: {comp_info.get('original_size_bytes', 0) / 1024:.2f} KB, Compressed: {comp_info.get('final_size_bytes', 0) / 1024:.2f} KB")

                except Exception as e:
                    print(f"Error processing item {batch_idx}-{i}: {e}")
                    import traceback
                    traceback.print_exc()

    # Calculate overall statistics
    avg_compression_ratio = 0
    avg_sparsity_ratio = 0

    if compression_stats['compression_ratios']:
        avg_compression_ratio = sum(compression_stats['compression_ratios']) / len(
            compression_stats['compression_ratios'])

    if compression_stats['sparsity_ratios']:
        avg_sparsity_ratio = sum(compression_stats['sparsity_ratios']) / len(compression_stats['sparsity_ratios'])

    # Prepare to save the distillation dataset
    total_items = len(distillation_data)
    print(f"\nSaving {total_items} {dataset_type} items to {output_path}...")
    print(
        f"Logits success rate: {success_count}/{success_count + failure_count} ({success_count / (success_count + failure_count) * 100:.1f}%)")

    # Calculate estimated file size
    estimated_jsonl_size_mb = compression_stats[
                                  'compressed_size_total'] / 1024 / 1024 * 1.1  # Add 10% for JSON overhead

    # Print compression statistics
    print(f"\nCompression Statistics:")
    for format_name, count in compression_stats['format_counts'].items():
        print(f"  Format: {format_name}, Count: {count}")
    print(f"  Original size total: {compression_stats['original_size_total'] / 1024 / 1024:.2f} MB")
    print(f"  Compressed size total: {compression_stats['compressed_size_total'] / 1024 / 1024:.2f} MB")
    print(f"  Estimated JSONL file size: {estimated_jsonl_size_mb:.2f} MB")
    print(
        f"  Overall compression ratio: {compression_stats['original_size_total'] / (compression_stats['compressed_size_total'] or 1):.2f}x")
    print(f"  Average compression ratio: {avg_compression_ratio:.2f}x")

    # Add compression stats to the file header
    # Remove sparsity-related stats from the file_stats dictionary
    file_stats = {
        'dataset_type': dataset_type,
        'item_count': total_items,
        'compression': {
            'original_size_mb': compression_stats['original_size_total'] / 1024 / 1024,
            'compressed_size_mb': compression_stats['compressed_size_total'] / 1024 / 1024,
            'estimated_file_size_mb': estimated_jsonl_size_mb,
            'overall_compression_ratio': compression_stats['original_size_total'] / (
                    compression_stats['compressed_size_total'] or 1),
            'avg_compression_ratio': avg_compression_ratio,
            'format_counts': compression_stats['format_counts']
        },
        'created_at': time.strftime("%Y-%m-%d %H:%M:%S"),
        'args': {
            'bits': args.compression_bits,
            'model_type': args.model_type,
            'fp16': args.fp16
        }
    }

    # Track file writing progress
    start_time = time.time()
    items_written = 0
    file_size_bytes = 0

    # Use JSON serialization with error handling and progress reporting
    with open(output_path, 'w') as f:
        # Write the header first
        header_json = json.dumps({"header": file_stats}) + '\n'
        f.write(header_json)
        file_size_bytes += len(header_json.encode('utf-8'))

        # Then write each data item
        for idx, item in enumerate(tqdm(distillation_data, desc=f"Writing {dataset_type} dataset to disk")):
            try:
                # First, convert any NumPy types to Python native types
                converted_item = convert_numpy_to_python(item)
                json_line = json.dumps(converted_item) + '\n'
                f.write(json_line)

                # Track progress
                items_written += 1
                file_size_bytes += len(json_line.encode('utf-8'))

                # Report progress for large datasets
                if total_items > 1000 and (idx + 1) % 1000 == 0:
                    elapsed = time.time() - start_time
                    items_per_sec = (idx + 1) / elapsed if elapsed > 0 else 0
                    estimated_total_time = total_items / items_per_sec if items_per_sec > 0 else 0
                    remaining_time = max(0, estimated_total_time - elapsed)

                    print(f"  Progress: {idx + 1}/{total_items} items ({(idx + 1) / total_items * 100:.1f}%), "
                          f"{items_per_sec:.1f} items/sec, "
                          f"file size: {file_size_bytes / 1024 / 1024:.2f} MB, "
                          f"remaining: {remaining_time / 60:.1f} minutes")

            except TypeError as e:
                print(f"Error serializing item {idx}: {e}")
                # Print the problematic keys and their types
                for k, v in item.items():
                    if k == "compressed_logits" and v is not None:
                        print(f"  compressed_logits keys: {list(v.keys())}")
                        for logit_k, logit_v in v.items():
                            print(f"    {logit_k}: {type(logit_v)}")
                    else:
                        print(f"  {k}: {type(v)}")

    # Final report
    elapsed = time.time() - start_time
    print(f"\n{dataset_type.capitalize()} distillation dataset created successfully!")
    print(f"  Total items written: {items_written}/{total_items}")
    print(f"  Actual file size: {file_size_bytes / 1024 / 1024:.2f} MB")
    print(f"  Write speed: {items_written / elapsed:.1f} items/sec")
    print(f"  Write time: {elapsed / 60:.1f} minutes")

    return distillation_data


def train_model(model, tokenizer, train_dataloader, val_dataloader, args):
    """Train the CodeT5 model on assertion generation task with metrics tracking"""

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Setup tensorboard if available
    try:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
        use_tensorboard = True
    except ImportError:
        print("TensorBoard not available. Install with pip install tensorboard for better logging.")
        use_tensorboard = False

    # Prepare optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Calculate total training steps
    num_epochs = args.epochs
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * num_epochs

    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )

    # Mixed precision training if requested
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None

    # Track metrics
    best_val_loss = float('inf')
    best_similarity = 0.0
    global_step = 0
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save training arguments
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Create metrics file
    metrics_file = os.path.join(args.output_dir, "training_metrics.csv")
    with open(metrics_file, "w") as f:
        f.write("epoch,global_step,train_loss,eval_loss,accuracy,similarity,f1,precision,recall\n")

    # Main training loop
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        epoch_loss = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass with optional mixed precision
            if args.fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / args.gradient_accumulation_steps

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scheduler.step()
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1
            else:
                # Standard forward and backward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / args.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            # Track loss
            batch_loss = loss.item() * args.gradient_accumulation_steps
            epoch_loss += batch_loss
            progress_bar.set_postfix({"loss": batch_loss})

            # Evaluate periodically
            if args.eval_steps > 0 and global_step > 0 and global_step % args.eval_steps == 0:
                val_loss, eval_results = evaluate_model(model, tokenizer, val_dataloader, device, args)

                # Log metrics
                if use_tensorboard:
                    tensorboard_writer.add_scalar("eval_loss", val_loss, global_step)
                    for metric, value in eval_results.items():
                        if isinstance(value, (int, float)):
                            tensorboard_writer.add_scalar(f"eval_{metric}", value, global_step)

                # Print results
                print(f"\nEvaluation at step {global_step}:")
                print(f"  Loss: {val_loss:.4f}")
                print(f"  Similarity: {eval_results['similarity_score_avg']:.4f}")
                print(f"  Accuracy: {eval_results['accuracy']:.4f}")
                print(f"  F1: {eval_results['f1']:.4f}")

                # Check for improvement
                improved = False
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    improved = True
                    print(f"  New best validation loss: {val_loss:.4f}")

                if eval_results['similarity_score_avg'] > best_similarity:
                    best_similarity = eval_results['similarity_score_avg']
                    improved = True
                    print(f"  New best similarity score: {best_similarity:.4f}")

                # Save if improved
                if improved:
                    model_dir = os.path.join(args.output_dir, "best_model")
                    os.makedirs(model_dir, exist_ok=True)
                    model.save_pretrained(model_dir)
                    tokenizer.save_pretrained(model_dir)
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                # Early stopping
                if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                    print(f"Early stopping after {epochs_without_improvement} evaluations without improvement")
                    break

                # Return to training mode
                model.train()

            # Save checkpoint
            if args.save_steps > 0 and global_step > 0 and global_step % args.save_steps == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)

        # End of epoch
        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s")
        print(f"  Average training loss: {avg_train_loss:.4f}")

        # Evaluate at the end of each epoch
        print(f"  Evaluating epoch {epoch + 1}...")
        val_loss, eval_results = evaluate_model(model, tokenizer, val_dataloader, device, args)
        val_losses.append(val_loss)

        # Log to tensorboard
        if use_tensorboard:
            tensorboard_writer.add_scalar("epoch_train_loss", avg_train_loss, epoch + 1)
            tensorboard_writer.add_scalar("epoch_val_loss", val_loss, epoch + 1)
            for metric, value in eval_results.items():
                if isinstance(value, (int, float)):
                    tensorboard_writer.add_scalar(f"epoch_eval_{metric}", value, epoch + 1)

        # Log to CSV
        with open(metrics_file, "a") as f:
            f.write(f"{epoch + 1},{global_step},{avg_train_loss:.6f},{val_loss:.6f},"
                    f"{eval_results['accuracy']:.6f},{eval_results['similarity_score_avg']:.6f},"
                    f"{eval_results['f1']:.6f},{eval_results['precision']:.6f},{eval_results['recall']:.6f}\n")

        # Print results
        print(f"  Validation loss: {val_loss:.4f}")
        print(f"  Similarity score: {eval_results['similarity_score_avg']:.4f}")
        print(f"  Accuracy: {eval_results['accuracy']:.4f}")
        print(f"  F1 score: {eval_results['f1']:.4f}")

        # Check for improvement
        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = True
            print(f"  New best validation loss: {val_loss:.4f}")

        if eval_results['similarity_score_avg'] > best_similarity:
            best_similarity = eval_results['similarity_score_avg']
            improved = True
            print(f"  New best similarity score: {best_similarity:.4f}")

        # Save if improved
        if improved:
            model_dir = os.path.join(args.output_dir, "best_model")
            os.makedirs(model_dir, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping after {epochs_without_improvement} epochs without improvement")
            break

        # Save final model
        final_model_dir = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)

    # Create distillation datasets if requested
    if args.create_distillation_dataset:
        # Create distillation dataset for validation data
        val_distillation_path = os.path.join(args.output_dir, "distillation_data_validation.jsonl")
        print(f"Creating validation distillation dataset at {val_distillation_path}...")
        create_distillation_dataset(model, tokenizer, val_dataloader, device, args, val_distillation_path, "validation")

        # Create distillation dataset for training data
        train_distillation_path = os.path.join(args.output_dir, "distillation_data_training.jsonl")
        print(f"Creating training distillation dataset at {train_distillation_path}...")
        create_distillation_dataset(model, tokenizer, train_dataloader, device, args, train_distillation_path, "training")

    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "loss_curves.png"))

    # Close tensorboard writer
    if use_tensorboard:
        tensorboard_writer.close()

    return model, tokenizer, best_val_loss


def evaluate_model(model, tokenizer, dataloader, device, args):
    """Evaluate model on dataloader"""
    model.eval()
    eval_loss = 0.0
    all_metrics = {
        "exact_matches": 0,
        "generated_count": 0,
        "reference_count": 0,
        "similarity_scores": [],
        "accuracy_scores": [],
        "f1_scores": []
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Track loss
            loss = outputs.loss
            eval_loss += loss.item()

            # Generate predictions for a subset of examples to save time
            subset_size = min(args.eval_batch_size, input_ids.size(0))
            subset_indices = np.random.choice(input_ids.size(0), subset_size, replace=False)

            # Generate for subset
            for idx in subset_indices:
                # Generate
                generated_ids = model.generate(
                    input_ids=input_ids[idx:idx + 1],
                    attention_mask=attention_mask[idx:idx + 1],
                    max_length=args.max_tgt_length,
                    num_beams=4,
                    early_stopping=True
                )

                # Decode prediction and reference
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                reference_text = batch["original_target"][idx]

                # Evaluate
                metrics = evaluate_assertions(generated_text, reference_text)

                # Update metrics
                all_metrics["exact_matches"] += metrics["exact_matches"]
                all_metrics["generated_count"] += metrics["generated_count"]
                all_metrics["reference_count"] += metrics["reference_count"]
                all_metrics["similarity_scores"].extend(metrics["similarity_scores"])
                all_metrics["accuracy_scores"].append(metrics["accuracy"])
                all_metrics["f1_scores"].append(metrics["f1"])

                # Display sample predictions occasionally
                if np.random.random() < 0.01:  # Show ~1% of predictions
                    print("\nExample evaluation:")
                    print(f"Reference: {reference_text[:100]}...")
                    print(f"Generated: {generated_text[:100]}...")
                    print(f"Metrics: Exact matches={metrics['exact_matches']}, "
                          f"Accuracy={metrics['accuracy']:.4f}, "
                          f"Similarity={metrics['similarity_score_avg']:.4f}")

    # Calculate overall metrics
    avg_loss = eval_loss / len(dataloader)

    # Handle empty metrics
    if not all_metrics["similarity_scores"]:
        return avg_loss, {
            "precision": 0, "recall": 0, "f1": 0, "accuracy": 0,
            "similarity_score_avg": 0
        }

    # Calculate aggregate metrics
    overall_precision = all_metrics["exact_matches"] / all_metrics["generated_count"] if all_metrics[
        "generated_count"] else 0
    overall_recall = all_metrics["exact_matches"] / all_metrics["reference_count"] if all_metrics[
        "reference_count"] else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (
                                                                                                              overall_precision + overall_recall) > 0 else 0

    # Average per-sample metrics
    avg_similarity = sum(all_metrics["similarity_scores"]) / len(all_metrics["similarity_scores"])
    avg_accuracy = sum(all_metrics["accuracy_scores"]) / len(all_metrics["accuracy_scores"]) if all_metrics[
        "accuracy_scores"] else 0
    avg_f1 = sum(all_metrics["f1_scores"]) / len(all_metrics["f1_scores"]) if all_metrics["f1_scores"] else 0

    eval_results = {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "accuracy": avg_accuracy,
        "similarity_score_avg": avg_similarity,
        "total_exact_matches": all_metrics["exact_matches"],
        "total_generated": all_metrics["generated_count"],
        "total_reference": all_metrics["reference_count"]
    }

    return avg_loss, eval_results


def extract_logits_from_codet5(model, tokenizer, input_ids, attention_mask, target_text, device, max_length=512,
                               use_fp16=False):
    """Extract logits from CodeT5 for a given input and target with optional FP16 support"""
    try:
        # Set up autocast for mixed precision if requested
        autocast_context = torch.cuda.amp.autocast() if use_fp16 else nullcontext()

        with autocast_context:
            # Get encoder outputs
            encoder_outputs = model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )

            # Tokenize target text
            target_tokens = tokenizer(
                target_text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(device)

            # Create shifted decoder inputs (teacher forcing)
            decoder_input_ids = model._shift_right(target_tokens.input_ids)

            # Get decoder outputs
            decoder_outputs = model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                return_dict=True
            )

            # Get logits
            logits = model.lm_head(decoder_outputs.last_hidden_state)

            # If using FP16, cast back to float32 for consistent processing
            if use_fp16:
                logits = logits.float()

            return logits
    except Exception as e:
        print(f"Error extracting logits: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Fine-tune CodeT5 for assertion generation")

    # Data args
    parser.add_argument("--data_path", type=str, required=True, help="Path to assertion dataset jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--model_name", type=str, default="Salesforce/codet5-base", help="Model name or path")
    parser.add_argument("--model_type", type=str, choices=["codet5", "codet5p"], default="codet5",
                        help="CodeT5 variant")
    parser.add_argument("--validation_split", type=float, default=0.1, help="Validation set split ratio")
    parser.add_argument("--max_src_length", type=int, default=1024, help="Max source sequence length")
    parser.add_argument("--max_tgt_length", type=int, default=512, help="Max target sequence length")

    # Training args
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Logging and saving args
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps (0 to eval only at epoch end)")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint steps (0 to save only best model)")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience (0 to disable)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to use (None = use all)")

    # Distillation dataset args
    parser.add_argument("--create_distillation_dataset", action="store_true",
                        help="Create distillation dataset after training")
    parser.add_argument("--compression_bits", type=int, default=16, choices=[4, 8, 16, 32],
                        help="Bits for logits compression")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    data = load_dataset(args.data_path, args.max_samples)
    print(f"Loaded {len(data)} examples")

    # Split into train and validation sets
    train_data, val_data = train_test_split(data, test_size=args.validation_split, random_state=args.seed)
    print(f"Training on {len(train_data)} examples, validating on {len(val_data)} examples")

    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    if args.model_type == "codet5":
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    else:  # codet5p
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    # Create datasets
    train_dataset = AssertionDataset(
        train_data,
        tokenizer,
        max_src_length=args.max_src_length,
        max_tgt_length=args.max_tgt_length
    )
    val_dataset = AssertionDataset(
        val_data,
        tokenizer,
        max_src_length=args.max_src_length,
        max_tgt_length=args.max_tgt_length
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Train model
    model, tokenizer, best_val_loss = train_model(
        model,
        tokenizer,
        train_dataloader,
        val_dataloader,
        args
    )

    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    print(f"Trained model and checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()