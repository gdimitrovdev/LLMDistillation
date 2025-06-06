import torch
from tqdm import tqdm

from evaluation.utils import check_java_parsability, evaluate_assertions, evaluate_assertions_with_codebleu_codet5_tokenizer


def evaluate_model(model, tokenizer, dataloader, device):
    """Evaluate model on dataloader"""
    model.eval()
    eval_loss = 0.0

    codebleu_scores = []
    ngram_match_scores = []
    weighted_ngram_match_scores = []
    syntax_match_scores = []
    dataflow_match_scores = []

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
                    codebleu_score_item = evaluate_assertions_with_codebleu_codet5_tokenizer(
                        reference_text.split("\n"),
                        generated_text.split("\n"),
                        tokenizer,
                    )
                    codebleu_scores.append(codebleu_score_item['codebleu'])
                    ngram_match_scores.append(codebleu_score_item['ngram_match_score'])
                    weighted_ngram_match_scores.append(codebleu_score_item['weighted_ngram_match_score'])
                    syntax_match_scores.append(codebleu_score_item['syntax_match_score'])
                    dataflow_match_scores.append(codebleu_score_item['dataflow_match_score'])
                except ZeroDivisionError:
                    codebleu_scores.append(0.0)
                    ngram_match_scores.append(0.0)
                    weighted_ngram_match_scores.append(0.0)
                    syntax_match_scores.append(0.0)
                    dataflow_match_scores.append(0.0)
                except Exception as e:
                    print(f"Error calculating CodeBLEU for an item: {e}")
                    codebleu_scores.append(0.0)
                    ngram_match_scores.append(0.0)
                    weighted_ngram_match_scores.append(0.0)
                    syntax_match_scores.append(0.0)
                    dataflow_match_scores.append(0.0)

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

    avg_codebleu = sum(codebleu_scores) / len(codebleu_scores) if codebleu_scores else 0.0
    avg_ngram = sum(ngram_match_scores) / len(ngram_match_scores) if ngram_match_scores else 0.0
    avg_weighted_ngram = sum(weighted_ngram_match_scores) / len(weighted_ngram_match_scores) if weighted_ngram_match_scores else 0.0
    avg_syntax_match = sum(syntax_match_scores) / len(syntax_match_scores) if syntax_match_scores else 0.0
    avg_dataflow_match = sum(dataflow_match_scores) / len(dataflow_match_scores) if dataflow_match_scores else 0.0

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
        "avg_codebleu_score": avg_codebleu,
        "avg_ngram_score": avg_ngram,
        "avg_weighted_ngram_score": avg_weighted_ngram,
        "avg_syntax_match_score": avg_syntax_match,
        "avg_dataflow_match_score": avg_dataflow_match,
        "total_assertion_blocks": total_assertion_blocks,
        "parsable_assertion_blocks": parsable_assertion_blocks,
        "parsability_rate": parsable_assertion_blocks / total_assertion_blocks,
    }

    print(f"  Validation loss: {avg_loss:.4f}")
    print(f"  Similarity score: {eval_results['similarity_score_avg']:.4f}")
    print(f"  Accuracy: {eval_results['accuracy']:.4f}")
    print(f"  F1 score: {eval_results['f1']:.4f}")
    print(f"  CodeBLEU score: {eval_results['avg_codebleu_score']:.4f}")
    print(f"    n-gram match score: {eval_results['avg_ngram_score']:.4f}")
    print(f"    Weighted n-gram match score: {eval_results['avg_weighted_ngram_score']:.4f}")
    print(f"    Syntax match score: {eval_results['avg_syntax_match_score']:.4f}")
    print(f"    Dataflow match score: {eval_results['avg_dataflow_match_score']:.4f}")
    print(f"  Parsability rate: {eval_results['parsability_rate']:.4f}")

    return avg_loss, eval_results
