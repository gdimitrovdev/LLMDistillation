import json
from tqdm import tqdm


def load_dataset(jsonl_path, max_samples=None, teacher_model="codet5"):
    """Load data from JSONL file with optional sample limit"""
    data = []
    total_lines = 0
    valid_lines = 0

    # First count lines
    with open(jsonl_path, 'r') as f:
        for _ in f:
            total_lines += 1
            if total_lines >= max_samples: break

    # Then load with progress bar
    with open(jsonl_path, 'r') as f:
        for line in tqdm(f, total=total_lines, desc="Loading dataset"):
            if line.strip():
                try:
                    entry = json.loads(line)
                    # Validate required fields are present
                    if 'focal_file' in entry and 'test_method_masked' in entry and 'original_target' in entry and 'model_type' in entry and entry['model_type'] == teacher_model:
                        data.append(entry)
                        valid_lines += 1
                        if max_samples and valid_lines >= max_samples:
                            break
                    else:
                        print("Warning: Skipping entry with missing fields")
                except json.JSONDecodeError:
                    print("Warning: Skipping invalid JSON line")

    return data