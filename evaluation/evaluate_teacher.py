import transformers
from transformers import RobertaTokenizer
from bert_score import score as bert_score_calculate

from data.load_dataset import load_dataset
from evaluation.utils_codeblue import evaluate_assertions_with_codebleu_codet5_tokenizer
from evaluation.utils import check_java_parsability, evaluate_assertions


transformers.logging.set_verbosity_error()


def evaluate_teacher(args):
    """Evaluate teacher model based on existing data"""
    val_data = load_dataset(args["data_path_validation"], args["max_samples_validation"])
    tokenizer = RobertaTokenizer.from_pretrained(args["teacher_model_name"])

    codebleu_scores = []
    ngram_match_scores = []
    weighted_ngram_match_scores = []
    syntax_match_scores = []
    dataflow_match_scores = []

    codebert_f1_scores = []
    CODE_MODEL_FOR_BERT_SCORE = "microsoft/graphcodebert-base"

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

        _, _, F1 = bert_score_calculate(
            cands=[generated_text],
            refs=[reference_text],
            model_type=CODE_MODEL_FOR_BERT_SCORE,
            lang="java",
            verbose=False,
            device=None,
            num_layers=12,
        )
        codebert_f1_scores.append(F1)

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

    avg_codebleu = sum(codebleu_scores) / len(codebleu_scores) if codebleu_scores else 0.0
    avg_ngram = sum(ngram_match_scores) / len(ngram_match_scores) if ngram_match_scores else 0.0
    avg_weighted_ngram = sum(weighted_ngram_match_scores) / len(weighted_ngram_match_scores) if weighted_ngram_match_scores else 0.0
    avg_syntax_match = sum(syntax_match_scores) / len(syntax_match_scores) if syntax_match_scores else 0.0
    avg_dataflow_match = sum(dataflow_match_scores) / len(dataflow_match_scores) if dataflow_match_scores else 0.0

    avg_codebert_f1 = sum(codebert_f1_scores) / len(codebert_f1_scores) if codebert_f1_scores else 0.0

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
        "avg_codebert_f1_score": avg_codebert_f1,
        "total_assertion_blocks": total_assertion_blocks,
        "parsable_assertion_blocks": parsable_assertion_blocks,
        "parsability_rate": parsable_assertion_blocks / total_assertion_blocks,
    }

    print(f"  Similarity score: {eval_results['similarity_score_avg']:.4f}")
    print(f"  Accuracy: {eval_results['accuracy']:.4f}")
    print(f"  F1 score: {eval_results['f1']:.4f}")
    print(f"  CodeBLEU score: {eval_results['avg_codebleu_score']:.4f}")
    print(f"    n-gram match score: {eval_results['avg_ngram_score']:.4f}")
    print(f"    Weighted n-gram match score: {eval_results['avg_weighted_ngram_score']:.4f}")
    print(f"    Syntax match score: {eval_results['avg_syntax_match_score']:.4f}")
    print(f"    Dataflow match score: {eval_results['avg_dataflow_match_score']:.4f}")
    print(f"  CodeBERTScore F1: {eval_results['avg_codebert_f1_score']}")
    print(f"  Parsability rate: {eval_results['parsability_rate']:.4f}")

    return eval_results
