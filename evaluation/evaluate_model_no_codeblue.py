import torch
from tqdm import tqdm

from evaluation.utils_no_codeblue import check_java_parsability, evaluate_assertions


def evaluate_model(model, tokenizer, dataloader, device):
    """Evaluate model on dataloader"""
    model.eval()
    eval_loss = 0.0

    parsable_assertion_blocks = 0
    total_assertion_blocks = 0

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

                total_assertion_blocks += 1

                if check_java_parsability(generated_text):
                    parsable_assertion_blocks += 1

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
        "total_reference": all_metrics["reference_count"],
        "total_assertion_blocks": total_assertion_blocks,
        "parsable_assertion_blocks": parsable_assertion_blocks,
        "parsability_rate": parsable_assertion_blocks / total_assertion_blocks,
    }

    print(f"  Validation loss: {avg_loss:.4f}")
    print(f"  Similarity score: {eval_results['similarity_score_avg']:.4f}")
    print(f"  Accuracy: {eval_results['accuracy']:.4f}")
    print(f"  F1 score: {eval_results['f1']:.4f}")
    print(f"  Parsability rate: {eval_results['parsability_rate']:.4f}")

    return avg_loss, eval_results
