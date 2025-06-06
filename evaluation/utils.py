import re
from difflib import SequenceMatcher
import javalang


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
    except Exception as e:
        print(e)
        print(final_assertion_code_for_parsing)
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
