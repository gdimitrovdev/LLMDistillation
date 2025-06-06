import json
import torch
from torch.utils.data import DataLoader
from transformers import (
    T5ForConditionalGeneration,
    RobertaTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from data.student_dataset import StudentDataset
from data.load_dataset import load_dataset
from evaluation.evaluate_model import evaluate_model, evaluate_assertions


def train_student_model(model, tokenizer, train_dataloader, val_dataloader, args):
    """Train the student model to match the teacher's output on the assertion generation task"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Setup tensorboard if available
    try:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_writer = SummaryWriter(log_dir=os.path.join(args["output_dir"], "tensorboard"))
        use_tensorboard = True
    except ImportError:
        use_tensorboard = False

    # Prepare optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args["learning_rate"])

    # Calculate total training steps
    if args["max_steps"] > 0:
        t_total = args["max_steps"]
        num_epochs = args["max_steps"] // (len(train_dataloader) // args["gradient_accumulation_steps"]) + 1
    else:
        t_total = len(train_dataloader) // args["gradient_accumulation_steps"] * args["epochs"]
        num_epochs = args["epochs"]

    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args["warmup_steps"],
        num_training_steps=t_total
    )

    # Mixed precision training if requested
    scaler = torch.cuda.amp.GradScaler() if args["fp16"] else None

    # Track metrics
    best_val_loss = float('inf')
    global_step = 0
    epochs_without_improvement = 0

    # For per-batch metrics
    batch_loss_window_size = args["batch_metrics_window"]
    batch_losses = []
    batch_accuracies = []
    batch_similarities = []
    eval_pool_size = args["batch_eval_pool"]  # Number of examples to evaluate in each batch

    # Create output directory
    os.makedirs(args["output_dir"], exist_ok=True)

    # Save training arguments
    with open(os.path.join(args["output_dir"], "training_args.json"), "w") as f:
        json.dump(args, f, indent=4)

    # Create metrics file
    metrics_file = os.path.join(args["output_dir"], "training_metrics.csv")
    with open(metrics_file, "w") as f:
        f.write("epoch,batch,global_step,loss,accuracy,similarity,lr,examples_per_second\n")

    # Create results file
    epochs_results_file = os.path.join(args["output_dir"], "epochs_evaluation_results.jsonl")

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
            teacher_logits = batch["teacher_logits"].to(device)

            # Forward pass with optional mixed precision
            if args["fp16"]:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss_ce = outputs.loss / args["gradient_accumulation_steps"]
                    student_logits = outputs.logits
                    
                    if student_logits.shape[1] != teacher_logits.shape[1]:
                        print("Student and teacher logits sizes do not match")
                        print(teacher_logits.shape[1])
                        print(student_logits.shape[1])
                        pass
                    
                    active_loss_mask = labels.view(-1) != -100 # Flattened mask

                    active_student_log_probs = torch.nn.functional.log_softmax(
                        student_logits.view(-1, student_logits.size(-1))[active_loss_mask] / args["distillation_temp"],
                        dim=-1
                    )
                    active_teacher_probs = torch.nn.functional.softmax(
                        teacher_logits.view(-1, teacher_logits.size(-1))[active_loss_mask] / args["distillation_temp"],
                        dim=-1
                    )

                    if active_student_log_probs.numel() > 0:
                        loss_fct_kl = torch.nn.KLDivLoss(reduction="batchmean")
                        loss_distill = loss_fct_kl(
                            active_student_log_probs,
                            active_teacher_probs
                        ) * (args["distillation_temp"] ** 2) # Scale by T^2
                    else:
                        loss_distill = torch.tensor(0.0, device=loss_ce.device, dtype=loss_ce.dtype)
                        
                    loss = args["alpha_ce"] * loss_ce + args["alpha_distil"] * loss_distill

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                if (batch_idx + 1) % args["gradient_accumulation_steps"] == 0:
                    # Unscales the gradients
                    scaler.unscale_(optimizer)

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args["max_grad_norm"])

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
                loss_ce = outputs.loss / args["gradient_accumulation_steps"]
                student_logits = outputs.logits
                    
                if student_logits.shape[1] != teacher_logits.shape[1]:
                    print("Student and teacher logits sizes do not match")
                    pass

                active_loss_mask = labels.view(-1) != -100 # Flattened mask

                active_student_log_probs = torch.nn.functional.log_softmax(
                    student_logits.view(-1, student_logits.size(-1))[active_loss_mask] / args["distillation_temp"],
                    dim=-1
                )
                active_teacher_probs = torch.nn.functional.softmax(
                    teacher_logits.view(-1, teacher_logits.size(-1))[active_loss_mask] / args["distillation_temp"],
                    dim=-1
                )

                if active_student_log_probs.numel() > 0:
                    loss_fct_kl = torch.nn.KLDivLoss(reduction="batchmean")
                    loss_distill = loss_fct_kl(
                        active_student_log_probs,
                        active_teacher_probs
                    ) * (args["distillation_temp"] ** 2) # Scale by T^2
                else:
                    loss_distill = torch.tensor(0.0, device=loss_ce.device, dtype=loss_ce.dtype)
                    
                loss = args["alpha_ce"] * loss_ce + args["alpha_distil"] * loss_distill

                # Backward pass
                loss.backward()

                if (batch_idx + 1) % args["gradient_accumulation_steps"] == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args["max_grad_norm"])

                    # Update weights
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            # Track the loss
            loss_value = loss.item() * args["gradient_accumulation_steps"]
            epoch_loss += loss_value
            batch_losses.append(loss_value)
            if len(batch_losses) > batch_loss_window_size:
                batch_losses.pop(0)

            # Calculate time per example
            batch_time = time.time() - batch_start_time
            examples_per_second = examples_in_batch / batch_time if batch_time > 0 else 0

            # Per-batch metrics: Generate predictions for a few examples to calculate accuracy
            if args["track_batch_metrics"] and batch_idx % args["batch_metrics_every"] == 0:
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
                        max_length=args["max_tgt_length"],
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
            if args["eval_steps"] > 0 and global_step % args["eval_steps"] == 0:
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
                    model_dir = os.path.join(args["output_dir"], "best_model")
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
                if 0 < args["early_stopping_patience"] <= epochs_without_improvement:
                    print(f"Early stopping after {epochs_without_improvement} evaluations without improvement")
                    break

                # Back to training mode
                model.train()

            # Save checkpoint
            if args["save_steps"] > 0 and global_step % args["save_steps"] == 0:
                checkpoint_dir = os.path.join(args["output_dir"], f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)

                # Save optimizer and scheduler
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))

            # Break if max steps reached
            if args["max_steps"] > 0 and global_step >= args["max_steps"]:
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
        print(f"  BLEU score: {eval_results['avg_bleu_score']:.4f}")
        print(f"  Parsability rate: {eval_results['parsability_rate']:.4f}")

        # Write results to result file
        epoch_summary = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "training_loss_avg": epoch_avg_loss,
            "validation_loss": val_loss,
            **eval_results
        }
        with open(epochs_results_file, "a", encoding="utf-8") as erf:
            erf.write(json.dumps(epoch_summary) + "\n")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  New best validation loss: {val_loss:.4f}")

            # Save model checkpoint
            model_dir = os.path.join(args["output_dir"], "best_model")
            os.makedirs(model_dir, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)

            # Reset patience counter
            epochs_without_improvement = 0
        else:
            # Increment patience counter
            epochs_without_improvement += 1

        # Early stopping
        if 0 < args["early_stopping_patience"] <= epochs_without_improvement:
            print(f"Early stopping after {epochs_without_improvement} epochs without improvement")
            break

    # Save final model
    final_model_dir = os.path.join(args["output_dir"], "final_model")
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
    plt.savefig(os.path.join(args["output_dir"], "loss_curves.png"))
    plt.close()

    return model, tokenizer, best_val_loss


def run_student_training(args):
    # Set random seed
    torch.manual_seed(args["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args["seed"])

    # Load training dataset
    print(f"Loading training dataset from {args['data_path_training']}...")
    if args["max_samples_training"] is not None:
        train_data = load_dataset(args["data_path_training"], args["max_samples_training"])
        print(f"Using first {len(train_data)} examples")
    else:
        train_data = load_dataset(args["data_path_training"])
        print(f"Loaded {len(train_data)} examples")
        
    # Load validation dataset
    print(f"Loading validation dataset from {args['data_path_validation']}...")
    if args["max_samples_validation"] is not None:
        val_data = load_dataset(args["data_path_validation"], args["max_samples_validation"])
        print(f"Using first {len(val_data)} examples")
    else:
        val_data = load_dataset(args["data_path_validation"])
        print(f"Loaded {len(val_data)} examples")

    print(f"Training on {len(train_data)} examples, validating on {len(val_data)} examples")

    # Load tokenizer and model
    if args["model_name"]:
        print(f"Loading model: {args['model_name']}")
        model = T5ForConditionalGeneration.from_pretrained(args["model_name"])
    else:
        print("Using custom model")
        model = args["model"]
        
    tokenizer = RobertaTokenizer.from_pretrained(args["teacher_model_name"])

    # Create datasets
    train_dataset = StudentDataset(
        train_data,
        tokenizer,
        max_src_length=args["max_src_length"],
        max_tgt_length=args["max_tgt_length"]
    )
    val_dataset = StudentDataset(
        val_data,
        tokenizer,
        max_src_length=args["max_src_length"],
        max_tgt_length=args["max_tgt_length"]
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args["num_workers"],
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args["eval_batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
        pin_memory=True
    )

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train model
    model, tokenizer, best_val_loss = train_student_model(
        model,
        tokenizer,
        train_dataloader,
        val_dataloader,
        args
    )

    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    print(f"Trained model and checkpoints saved to {args['output_dir']}")
