import torch
from tqdm import tqdm
import re
from difflib import SequenceMatcher
import nltk
import javalang

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def check_java_parsability(generated_assertion_block_str):
    """
    Checks if a block of generated assertion strings is parsable by javalang
    by wrapping it in a minimal class and method structure.
    This does NOT check for semantic correctness (e.g., undeclared variables)
    or import resolution. It primarily checks statement-level syntax.
    """
    processed_assertion_code = []
    for line in generated_assertion_block_str.strip().split('\n'):
        line = line.strip()
        if line:
            processed_assertion_code.append(line)
    
    final_assertion_code_for_parsing = "\n            ".join(processed_assertion_code)
    if not final_assertion_code_for_parsing.strip():
        return True

    # Create a minimal valid Java structure to wrap the assertions
    code_to_parse = f"""
    // No package declaration needed for javalang.parse.parse if it's just a snippet.
    // No explicit imports are provided; javalang will treat unknown types as identifiers.
    // It checks if 'identifier.method(...);' or 'method(...);' is syntactically valid.

    class TemporaryParsingWrapperClass {{
        public void temporaryTestMethod() {{
            // Generated assertions go here
            {final_assertion_code_for_parsing}
        }}
    }}
    """
    try:
        javalang.parse.parse(code_to_parse)
        return True
    except javalang.parser.JavaSyntaxError:
        return False


def calculate_similarity(reference, candidate):
    """Calculate string similarity using SequenceMatcher"""
    return SequenceMatcher(None, reference, candidate).ratio()


def normalize_assertion(assertion):
    """Normalize assertion text for more reliable comparison"""
    # Remove whitespace
    assertion = re.sub(r'\s+', ' ', assertion).strip()

    # Remove variable names in certain cases
    assertion = re.sub(r'assertEquals\(\s*[^,]+,\s*([^)]+)\)', r'assertEquals(VALUE, \1)', assertion)

    # Normalize assertion method names
    assertion = re.sub(r'assert(Equals|That|True|False)', r'assert\1', assertion, flags=re.IGNORECASE)

    return assertion


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


def evaluate_model(model, tokenizer, dataloader, device):
    """Evaluate model on dataloader"""
    model.eval()
    eval_loss = 0.0

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

    # Calculate overall metrics
    avg_loss = eval_loss / len(dataloader)

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

    return avg_loss, eval_results
