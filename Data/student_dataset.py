from torch.utils.data import Dataset

from data_generation.compress import decompress_logits
from data_generation.train_codet5_assertions import clean_assertion_placeholders


class StudentDataset(Dataset):
    """Dataset for the student model"""

    def __init__(self, data, tokenizer, max_src_length=1024, max_tgt_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Extract data fields
        focal_method = item['focal_method']
        test_method_masked = item['test_method_masked']
        original_target = '\n'.join(item['assertions'])

        cleaned_focal_method = clean_assertion_placeholders(focal_method) if focal_method else ""
        cleaned_test_method = clean_assertion_placeholders(test_method_masked)

        # First, tokenize just the test method to determine its token length
        test_method_tokens = self.tokenizer(
            f"TEST METHOD:\n{cleaned_test_method}",
            add_special_tokens=True,
            truncation=True,  # Add truncation here
            max_length=self.max_src_length,  # Use max_src_length as the limit
            return_tensors="pt"
        )
        test_method_length = test_method_tokens.input_ids.size(1)

        # If test method already exceeds limit (rare but possible), we must truncate it
        if test_method_length >= self.max_src_length - 10:  # Leave room for special tokens
            # Just keep the test method, already truncated
            input_text = f"TEST METHOD:\n{self.tokenizer.decode(test_method_tokens.input_ids[0], skip_special_tokens=True)}"
        else:
            # Determine how much space we have left for the focal file
            space_for_focal = self.max_src_length - test_method_length - 20  # Reserve tokens for prefix and special tokens

            # Format input text based on available space
            if space_for_focal <= 0 or not cleaned_focal_method:
                # Not enough space - use only test method
                input_text = f"TEST METHOD:\n{cleaned_test_method}"
            else:
                # Tokenize focal file to check its length, with explicit truncation
                focal_tokens = self.tokenizer(
                    cleaned_focal_method,
                    add_special_tokens=False,
                    truncation=True,  # Add truncation here
                    max_length=space_for_focal,  # Limit to available space
                    return_tensors="pt"
                )

                # Create combined input with truncated focal file if needed
                truncated_focal = self.tokenizer.decode(focal_tokens.input_ids[0], skip_special_tokens=True)
                input_text = f"FOCAL METHOD:\n{truncated_focal}\n\nTEST METHOD:\n{cleaned_test_method}"

        # Apply strict truncation at the tokenizer level
        source_encoding = self.tokenizer(
            input_text,
            max_length=self.max_src_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            original_target,
            max_length=self.max_tgt_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Double-check lengths and force truncation if needed (safety check)
        if source_encoding["input_ids"].size(1) > self.max_src_length:
            source_encoding["input_ids"] = source_encoding["input_ids"][:, :self.max_src_length]
            source_encoding["attention_mask"] = source_encoding["attention_mask"][:, :self.max_src_length]

        if target_encoding["input_ids"].size(1) > self.max_tgt_length:
            target_encoding["input_ids"] = target_encoding["input_ids"][:, :self.max_tgt_length]

        input_ids = source_encoding["input_ids"].squeeze()
        attention_mask = source_encoding["attention_mask"].squeeze()
        labels = target_encoding["input_ids"].squeeze()

        # Replace padding token id with -100 so it's ignored in loss computation
        labels[labels == self.tokenizer.pad_token_id] = -100

        teacher_logits = decompress_logits(item["compressed_logits"]).squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "original_input": input_text,
            "original_target": original_target,
            "teacher_logits": teacher_logits,
            "idx": idx,
        }
