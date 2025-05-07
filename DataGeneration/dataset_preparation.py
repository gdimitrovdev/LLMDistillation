import re
import json
import os
import argparse
from typing import List, Dict, Any, Tuple
from tqdm import tqdm


def extract_assertions(test_method: str) -> Tuple[str, List[str]]:
    """
    Extract assertion statements from a test method and return both the
    assertions and the test method with assertions removed.

    Args:
        test_method: Java test method code as a string

    Returns:
        Tuple containing:
        - test_method_without_assertions: The test code with assertions removed
        - assertions: List of extracted assertion statements
    """
    # Regex to match assert statements (handles multi-line assertions too)
    assert_pattern = re.compile(r'\bassert\w+\s*\([^;]*?;', re.DOTALL)

    # Find all assertion statements
    assertion_matches = list(assert_pattern.finditer(test_method))
    assertions = [match.group(0) for match in assertion_matches]

    # Remove assertions from the test method
    test_method_without_assertions = test_method
    for match in reversed(assertion_matches):  # Process in reverse to maintain correct positions
        start, end = match.span()
        test_method_without_assertions = test_method_without_assertions[:start] + test_method_without_assertions[end:]

    # Clean up the test method (remove extra whitespace, etc.)
    test_method_without_assertions = re.sub(r'\n\s*\n+', '\n\n', test_method_without_assertions)
    test_method_without_assertions = test_method_without_assertions.strip()

    return test_method_without_assertions, assertions


def extract_method_under_test(test_method: str) -> str:
    """
    Extract the likely method under test based on the test method name and content.

    Args:
        test_method: Java test method code

    Returns:
        Name of the method being tested (or empty string if cannot determine)
    """
    # Extract test method name
    method_name_match = re.search(r'(?:public|private|protected)?\s+(?:void|[\w<>[\]]+)\s+(\w+)\s*\(', test_method)
    if not method_name_match:
        return ""

    test_method_name = method_name_match.group(1)

    # Remove "test" prefix/suffix if present
    method_under_test = re.sub(r'^test|Test$', '', test_method_name)

    # Look for method calls in the test
    method_calls = re.findall(r'(\w+)\s*\([^)]*\)', test_method)

    # Filter out common assertion method names and the test method itself
    common_methods = {'assertEquals', 'assertTrue', 'assertFalse', 'assertNotNull', 'assertNull',
                      'assertSame', 'assertNotSame', 'assertThat', 'assertThrows', 'assert'}
    possible_methods = [m for m in method_calls if m not in common_methods and m != test_method_name]

    # If we found potential methods under test, return the most frequent one
    if possible_methods:
        from collections import Counter
        counter = Counter(possible_methods)
        return counter.most_common(1)[0][0]

    return method_under_test if method_under_test else ""


def prepare_dataset(input_jsonl: str, output_jsonl: str):
    """
    Process the input JSONL file and create a structured dataset for CodeT5 training.

    Args:
        input_jsonl: Path to input JSONL file
        output_jsonl: Path to output JSONL file for the prepared dataset
    """
    processed_entries = []

    # First, count number of lines for the progress bar
    line_count = 0
    with open(input_jsonl, 'r') as f:
        for line in f:
            if line.strip():
                line_count += 1

    print(f"Found {line_count} lines in input file.")

    # Process each line with a progress bar
    from tqdm import tqdm
    with open(input_jsonl, 'r') as f:
        for line_number, line in tqdm(enumerate(f, 1), total=line_count, desc="Processing test methods"):
            if not line.strip():
                continue

            try:
                entry = json.loads(line)

                # Skip if required fields are missing
                if 'test_method' not in entry or 'focal_file' not in entry:
                    continue

                # Extract assertions and test case without assertions
                test_without_assertions, assertions = extract_assertions(entry['test_method'])

                # Skip if no assertions found
                if not assertions:
                    continue

                # Try to identify the method under test
                method_under_test = extract_method_under_test(entry['test_method'])

                # Create structured entry for the new dataset
                processed_entry = {
                    'repository': entry.get('repository', ''),
                    'focal_file': entry['focal_file'],  # Code under test
                    'test_method_original': entry['test_method'],  # Original test with assertions
                    'test_method_masked': test_without_assertions,  # Test with assertions removed
                    'assertions': assertions,  # List of assertions to predict
                    'method_under_test': method_under_test,  # Name of the method being tested
                }

                processed_entries.append(processed_entry)

            except json.JSONDecodeError:
                tqdm.write(f"Invalid JSON on line {line_number}, skipping.")
            except Exception as e:
                tqdm.write(f"Error processing line {line_number}: {str(e)}")

    # Write the processed entries to the output file
    print(f"Writing {len(processed_entries)} entries to output file...")
    with open(output_jsonl, 'w') as f:
        for entry in tqdm(processed_entries, desc="Writing output"):
            f.write(json.dumps(entry) + '\n')

    print(f"Successfully processed {len(processed_entries)} valid entries.")
    return processed_entries


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Prepare dataset for test assertion generation")
    parser.add_argument("--input_path", type=str,
                        default="/Users/apanichella/Downloads/data/prepared-repositories.jsonl",
                        help="Path to input JSONL file with Java repositories")
    parser.add_argument("--output_path", type=str,
                        default="/Users/apanichella/Desktop/CISELab/tiny-but-mighty/Distillation/data/assertion_dataset.jsonl",
                        help="Path to output JSONL file for processed dataset")
    parser.add_argument("--show_example", action="store_true",
                        help="Show an example entry from the processed dataset")
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Process the dataset
    processed_entries = prepare_dataset(args.input_path, args.output_path)

    # Print stats
    total_assertions = sum(len(entry['assertions']) for entry in processed_entries)
    print("\n=== Dataset Statistics ===")
    print(f"Total test methods processed: {len(processed_entries)}")
    print(f"Total assertions extracted: {total_assertions}")
    print(f"Average assertions per test method: {total_assertions / len(processed_entries):.2f}")

    # Print an example if requested
    if args.show_example and processed_entries:
        first_entry = processed_entries[0]
        print("\n=== Example Entry ===")
        print(f"Original Test Method:\n{first_entry['test_method_original'][:500]}...\n")
        print(f"Test Method without Assertions:\n{first_entry['test_method_masked'][:500]}...\n")
        print(f"Assertions to Predict:")
        for i, assertion in enumerate(first_entry['assertions'], 1):
            print(f"{i}. {assertion}")
        print(f"\nMethod Under Test: {first_entry['method_under_test']}")


if __name__ == "__main__":
    main()