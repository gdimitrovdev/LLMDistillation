#!/usr/bin/env python
"""
Script to fine-tune a CodeT5 model on Java test assertion generation.
Shows per-batch metrics including loss and accuracy.
"""

import argparse
import json
import os
import re
import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    T5ForConditionalGeneration,
    RobertaTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from difflib import SequenceMatcher
import time


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
            "original_target": target_text
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


def train_teacher_model(model, tokenizer, train_dataloader, val_dataloader, args):
    """Train the teacher model on assertion generation task"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Setup tensorboard if available
    try:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
        use_tensorboard = True
    except ImportError:
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
    if args.max_steps > 0:
        t_total = args.max_steps
        num_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
        num_epochs = args.epochs

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
    global_step = 0
    epochs_without_improvement = 0

    # For per-batch metrics
    batch_loss_window_size = args.batch_metrics_window
    batch_losses = []
    batch_accuracies = []
    batch_similarities = []
    eval_pool_size = args.batch_eval_pool  # Number of examples to evaluate in each batch

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save training arguments
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Create metrics file
    metrics_file = os.path.join(args.output_dir, "training_metrics.csv")
    with open(metrics_file, "w") as f:
        f.write("epoch,batch,global_step,loss,accuracy,similarity,lr,examples_per_second\n")

    # Main training loop
    print(f"Starting training for {num_epochs} epochs...")
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        examples_processed = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            batch_start_time = time.time()
            examples_in_batch = len(batch["input_ids"])
            examples_processed += examples_in_batch

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
                    # Unscales the gradients
                    scaler.unscale_(optimizer)

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    # Update weights
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
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    # Update weights
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            # Track the loss
            loss_value = loss.item() * args.gradient_accumulation_steps
            epoch_loss += loss_value
            batch_losses.append(loss_value)
            if len(batch_losses) > batch_loss_window_size:
                batch_losses.pop(0)

            # Calculate time per example
            batch_time = time.time() - batch_start_time
            examples_per_second = examples_in_batch / batch_time if batch_time > 0 else 0

            # Per-batch metrics: Generate predictions for a few examples to calculate accuracy
            if args.track_batch_metrics and batch_idx % args.batch_metrics_every == 0:
                # Sample some examples from batch to evaluate
                eval_indices = np.random.choice(
                    range(len(input_ids)),
                    size=min(eval_pool_size, len(input_ids)),
                    replace=False
                )

                # Switch to eval mode temporarily
                model.eval()
                with torch.no_grad():
                    # Generate predictions for sampled examples
                    sampled_input_ids = input_ids[eval_indices]
                    sampled_attention_mask = attention_mask[eval_indices]

                    generated_ids = model.generate(
                        input_ids=sampled_input_ids,
                        attention_mask=sampled_attention_mask,
                        max_length=args.max_tgt_length,
                        num_beams=4,
                        early_stopping=True
                    )

                    # Calculate accuracy and similarity
                    batch_accuracy = 0
                    batch_similarity = 0

                    for i, idx in enumerate(eval_indices):
                        generated_text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                        reference_text = batch["original_target"][idx]

                        try:
                            metrics = evaluate_assertions(generated_text, reference_text)
                            batch_accuracy += metrics["accuracy"]
                            batch_similarity += metrics["similarity_score_avg"]
                        except Exception as e:
                            print(f"Error evaluating assertion: {e}")

                    # Average the metrics
                    batch_accuracy /= len(eval_indices) if eval_indices else 1
                    batch_similarity /= len(eval_indices) if eval_indices else 1

                # Switch back to train mode
                model.train()

                # Track metrics
                batch_accuracies.append(batch_accuracy)
                batch_similarities.append(batch_similarity)
                if len(batch_accuracies) > batch_loss_window_size:
                    batch_accuracies.pop(0)
                if len(batch_similarities) > batch_loss_window_size:
                    batch_similarities.pop(0)

                # Calculate moving averages
                avg_loss = sum(batch_losses) / len(batch_losses)
                avg_accuracy = sum(batch_accuracies) / len(batch_accuracies) if batch_accuracies else 0
                avg_similarity = sum(batch_similarities) / len(batch_similarities) if batch_similarities else 0

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": avg_loss,
                    "accuracy": avg_accuracy,
                    "similarity": avg_similarity,
                    "ex/s": f"{examples_per_second:.1f}"
                })

                # Log to tensorboard
                if use_tensorboard:
                    tensorboard_writer.add_scalar("batch_loss", avg_loss, global_step)
                    tensorboard_writer.add_scalar("batch_accuracy", avg_accuracy, global_step)
                    tensorboard_writer.add_scalar("batch_similarity", avg_similarity, global_step)
                    tensorboard_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tensorboard_writer.add_scalar("examples_per_second", examples_per_second, global_step)

                # Log to CSV
                with open(metrics_file, "a") as f:
                    f.write(
                        f"{epoch + 1},{batch_idx + 1},{global_step},{avg_loss:.6f},{avg_accuracy:.6f},{avg_similarity:.6f},{scheduler.get_last_lr()[0]:.8f},{examples_per_second:.2f}\n")
            else:
                # Just update with loss
                avg_loss = sum(batch_losses) / len(batch_losses)
                progress_bar.set_postfix({
                    "loss": avg_loss,
                    "ex/s": f"{examples_per_second:.1f}"
                })

            # Evaluate periodically
            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                val_loss, eval_results = evaluate_model(model, tokenizer, val_dataloader, device)

                # Log to tensorboard
                if use_tensorboard:
                    tensorboard_writer.add_scalar("eval_loss", val_loss, global_step)
                    for metric, value in eval_results.items():
                        if isinstance(value, (int, float)):
                            tensorboard_writer.add_scalar(f"eval_{metric}", value, global_step)

                # Print evaluation results
                print(f"\nEvaluation at step {global_step}:")
                print(f"  Validation loss: {val_loss:.4f}")
                print(f"  Similarity score: {eval_results['similarity_score_avg']:.4f}")
                print(f"  Accuracy: {eval_results['accuracy']:.4f}")
                print(f"  F1 score: {eval_results['f1']:.4f}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"  New best validation loss: {val_loss:.4f}")

                    # Save model checkpoint
                    model_dir = os.path.join(args.output_dir, "best_model")
                    os.makedirs(model_dir, exist_ok=True)
                    model.save_pretrained(model_dir)
                    tokenizer.save_pretrained(model_dir)

                    # Save optimizer and scheduler
                    torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(model_dir, "scheduler.pt"))

                    # Reset patience counter
                    epochs_without_improvement = 0
                else:
                    # Increment patience counter
                    epochs_without_improvement += 1

                # Early stopping
                if 0 < args.early_stopping_patience <= epochs_without_improvement:
                    print(f"Early stopping after {epochs_without_improvement} evaluations without improvement")
                    break

                # Back to training mode
                model.train()

            # Save checkpoint
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)

                # Save optimizer and scheduler
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))

            # Break if max steps reached
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        # Calculate average loss for the epoch
        epoch_avg_loss = epoch_loss / len(train_dataloader)
        train_losses.append(epoch_avg_loss)

        # Calculate epoch time and speed
        epoch_time = time.time() - epoch_start_time
        examples_per_second = examples_processed / epoch_time if epoch_time > 0 else 0

        print(f"\nEpoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s ({examples_per_second:.2f} examples/s)")
        print(f"  Average training loss: {epoch_avg_loss:.4f}")

        # Evaluate at the end of each epoch
        print(f"  Evaluating epoch {epoch + 1}...")
        val_loss, eval_results = evaluate_model(model, tokenizer, val_dataloader, device)
        val_losses.append(val_loss)

        # Log to tensorboard
        if use_tensorboard:
            tensorboard_writer.add_scalar("epoch_train_loss", epoch_avg_loss, epoch + 1)
            tensorboard_writer.add_scalar("epoch_val_loss", val_loss, epoch + 1)
            for metric, value in eval_results.items():
                if isinstance(value, (int, float)):
                    tensorboard_writer.add_scalar(f"epoch_eval_{metric}", value, epoch + 1)

        # Print evaluation results
        print(f"  Validation loss: {val_loss:.4f}")
        print(f"  Similarity score: {eval_results['similarity_score_avg']:.4f}")
        print(f"  Accuracy: {eval_results['accuracy']:.4f}")
        print(f"  F1 score: {eval_results['f1']:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  New best validation loss: {val_loss:.4f}")

            # Save model checkpoint
            model_dir = os.path.join(args.output_dir, "best_model")
            os.makedirs(model_dir, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)

            # Reset patience counter
            epochs_without_improvement = 0
        else:
            # Increment patience counter
            epochs_without_improvement += 1

        # Early stopping
        if 0 < args.early_stopping_patience <= epochs_without_improvement:
            print(f"Early stopping after {epochs_without_improvement} epochs without improvement")
            break

    # Save final model
    final_model_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    # Close tensorboard writer
    if use_tensorboard:
        tensorboard_writer.close()

    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "loss_curves.png"))
    plt.close()

    return model, tokenizer, best_val_loss


def evaluate_model(model, tokenizer, dataloader, device):
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

            # Get loss
            loss = outputs.loss
            eval_loss += loss.item()

            # Generate predictions for evaluation
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )

            # Decode and evaluate
            for i in range(len(input_ids)):
                generated_text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                reference_text = batch["original_target"][i]

                try:
                    metrics = evaluate_assertions(generated_text, reference_text)

                    # Update metrics
                    all_metrics["exact_matches"] += metrics["exact_matches"]
                    all_metrics["generated_count"] += metrics["generated_count"]
                    all_metrics["reference_count"] += metrics["reference_count"]
                    all_metrics["similarity_scores"].extend(metrics["similarity_scores"])
                    all_metrics["accuracy_scores"].append(metrics["accuracy"])
                    all_metrics["f1_scores"].append(metrics["f1"])
                except Exception as e:
                    print(f"Error evaluating assertion: {e}")

    # Calculate overall metrics
    avg_loss = eval_loss / len(dataloader)

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

    eval_results = {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "accuracy": overall_accuracy,
        "avg_per_sample_accuracy": avg_per_sample_accuracy,
        "similarity_score_avg": avg_similarity,
        "avg_per_sample_f1": avg_f1,
        "total_exact_matches": all_metrics["exact_matches"],
        "total_generated": all_metrics["generated_count"],
        "total_reference": all_metrics["reference_count"]
    }

    return avg_loss, eval_results


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CodeT5 for assertion generation")

    # Data args
    parser.add_argument("--data_path", type=str, required=True, help="Path to assertion dataset jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--model_name", type=str, default="Salesforce/codet5-small", help="Model name or path")
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
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps (-1 for full epochs)")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Batch metrics args
    parser.add_argument("--track_batch_metrics", action="store_true", help="Track accuracy and similarity per batch")
    parser.add_argument("--batch_metrics_every", type=int, default=1, help="Calculate batch metrics every N batches")
    parser.add_argument("--batch_metrics_window", type=int, default=50, help="Window size for moving average")
    parser.add_argument("--batch_eval_pool", type=int, default=4, help="Examples to evaluate per batch")

    # Logging and saving args
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps (0 to eval only at epoch end)")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint steps (0 to save only best model)")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience (0 to disable)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use (None = use all)")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    data = load_dataset(args.data_path)
    if args.max_samples is not None:
        original_data = len(data)
        data = data[:args.max_samples]
        print(f"Using first {len(data)} examples out of total {original_data}")
    else:
        print(f"Loaded {len(data)} examples")

    # Split into train and validation sets
    train_data, val_data = train_test_split(data, test_size=args.validation_split, random_state=args.seed)
    print(f"Training on {len(train_data)} examples, validating on {len(val_data)} examples")

    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
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

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train model
    model, tokenizer, best_val_loss = train_teacher_model(
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