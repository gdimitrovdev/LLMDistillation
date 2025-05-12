from torch.utils.data import Dataset

from Data.decompression_test import decompress_tensor_optimized


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

        # Construct input: We combine focal code, test method without assertions
        input_text = f"FOCAL CODE:\n{item['focal_file']}\n\nTEST METHOD:\n{item['test_method_masked']}"

        # Target: The assertions that need to be generated
        target_text = "\n".join(item['assertions'])

        # Tokenize inputs
        source_encoding = self.tokenizer(
            input_text,
            max_length=self.max_src_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize targets
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_tgt_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = source_encoding["input_ids"].squeeze()
        attention_mask = source_encoding["attention_mask"].squeeze()
        labels = target_encoding["input_ids"].squeeze()

        # Replace padding token id with -100 so it's ignored in loss computation
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        decompressed_teacher_logits = decompress_tensor_optimized(item['teacher_logits'])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "original_input": input_text,
            "original_target": target_text,
            "idx": idx,
            "teacher_logits": decompressed_teacher_logits,
        }