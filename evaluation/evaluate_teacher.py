from transformers import RobertaTokenizer
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from data.load_dataset import load_dataset
from evaluation.utils import check_java_parsability, evaluate_assertions


def evaluate_teacher(args):
    """Evaluate teacher model based on existing data"""
    val_data = load_dataset(args["data_path_validation"], args["max_samples_validation"])
    tokenizer = RobertaTokenizer.from_pretrained(args["teacher_model_name"])

    item_bleu_scores = []
    chencherry = SmoothingFunction()

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

    for item in val_data:
        generated_text = "\n".join(item["predicted_assertions"])
        reference_text = "\n".join(item["assertions"])

        total_assertion_blocks += 1

        if check_java_parsability(generated_text):
            parsable_assertion_blocks += 1

        hypothesis_tokens = tokenizer.tokenize(generated_text.lower())
        reference_tokens_list = [tokenizer.tokenize(reference_text.lower())]

        try:
            bleu_score_item = sentence_bleu(
                references=reference_tokens_list,
                hypothesis=hypothesis_tokens,
                smoothing_function=chencherry.method4
            )
            item_bleu_scores.append(bleu_score_item)
        except ZeroDivisionError:
            item_bleu_scores.append(0.0) # Empty hypothesis or no overlap
        except Exception as e:
            print(f"Error calculating BLEU for an item: {e}")
            item_bleu_scores.append(0.0)

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

    avg_bleu = sum(item_bleu_scores) / len(item_bleu_scores) if item_bleu_scores else 0.0

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
        "avg_bleu_score": avg_bleu,
        "total_assertion_blocks": total_assertion_blocks,
        "parsable_assertion_blocks": parsable_assertion_blocks,
        "parsability_rate": parsable_assertion_blocks / total_assertion_blocks,
    }

    print(f"  Similarity score: {eval_results['similarity_score_avg']:.4f}")
    print(f"  Accuracy: {eval_results['accuracy']:.4f}")
    print(f"  F1 score: {eval_results['f1']:.4f}")
    print(f"  BLEU score: {eval_results['avg_bleu_score']:.4f}")
    print(f"  Parsability rate: {eval_results['parsability_rate']:.4f}")

    return eval_results
