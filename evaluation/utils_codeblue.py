from codebleu import calc_codebleu

from evaluation.utils import normalize_assertions


def evaluate_assertions_with_codebleu_codet5_tokenizer(
    ground_truth_assertions,
    generated_assertions,
    tokenizer,
):
    """
    Calculates CodeBLEU scores for a list of ground truth and generated Java assertions,
    using RobertaTokenizer from CodeT5 for the n-gram components.

    Args:
        ground_truth_assertions (list[str]): A list of reference assertion strings.
        generated_assertions (list[str]): A list of generated assertion strings.
        tokenizer (obj): The CodeT5 tokenizer.

    Returns:
        dict: A dictionary containing overall CodeBLEU score and scores for
              its components (ngram_match, weighted_ngram_match, syntax_match, dataflow_match).
    """
    normalized_generated, normalized_reference = normalize_assertions(generated_assertions, ground_truth_assertions)

    if len(normalized_generated) != len(normalized_reference):
        print("Number of assertions should be the same")
        return {
            "codebleu": 0.0,
            "ngram_match_score": 0.0,
            "weighted_ngram_match_score": 0.0,
            "syntax_match_score": 0.0,
            "dataflow_match_score": 0.0,
        }

    def custom_code_tokenizer(code_string):
        token_ids = tokenizer.encode(code_string, add_special_tokens=False)
        tokens = [tokenizer.decode(token_id) for token_id in token_ids]
        return tokens
    
    references_formatted = [[ref] for ref in normalized_reference]

    results = calc_codebleu(
        references=references_formatted,
        predictions=normalized_generated,
        lang="java",
        weights=(0.25, 0.25, 0.25, 0.25),
        tokenizer=custom_code_tokenizer,
    )

    return results
