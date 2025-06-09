import os
import torch
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader
import json

from data.load_dataset import load_dataset
from data.student_dataset import StudentDataset
from evaluation.evaluate_model import evaluate_model


def evaluate_after_epochs(model, output_file, code_dir=None):
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
    TEACHER_MODEL_TYPE = "codet5"
    BATCH_SIZE = 4
    MAX_SRC_LENGTH = 1024
    MAX_TGT_LENGTH = 512
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(DEVICE)

    relative_data_path = "data_generation/data/codet5/distillation_data_validation.jsonl"
    VALIDATION_DATA_PATH = relative_data_path

    eval_data = load_dataset(VALIDATION_DATA_PATH, max_samples=500, teacher_model=TEACHER_MODEL_TYPE)
    eval_dataset = StudentDataset(
        data=eval_data,
        tokenizer=tokenizer,
        max_src_length=MAX_SRC_LENGTH,
        max_tgt_length=MAX_TGT_LENGTH
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    _, results = evaluate_model(model, tokenizer, eval_dataloader, DEVICE)

    with open(output_file, "w", encoding="utf-8") as erf:
        erf.write(json.dumps(results) + "\n")
