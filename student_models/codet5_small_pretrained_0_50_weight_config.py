import os

def codet5_small_pretrained_0_50_weight_config(base_path):
    return {
        # --- Data Args ---
        "data_path_training": os.path.join(base_path, "data_generation/data/codet5/distillation_data_training.jsonl"),
        "data_path_validation": os.path.join(base_path, "data_generation/data/codet5/distillation_data_validation.jsonl"),
        "output_dir": os.path.join(base_path, "output_models/student_model_output_codet5_small_pretrained_0_50_weight"),
        "teacher_model_name": "Salesforce/codet5-base",
        "model_name": "Salesforce/codet5-small",
        "model": None, # Used for custom models
        "max_src_length": 1024,
        "max_tgt_length": 512,

        # --- Distillation Args ---
        "distillation_temp": 2.0,
        "alpha_ce": 0.5,            # Weight for student's own Cross-Entropy loss
        "alpha_distil": 0.5,        # Weight for distillation loss (e.g., KL divergence)

        # --- Training Args ---
        "epochs": 10,
        "batch_size": 4,
        "eval_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
        "max_steps": -1,
        "fp16": True,
        "seed": 42,

        # --- Batch Metrics Args (Optional, can simplify by removing) ---
        "track_batch_metrics": False,
        "batch_metrics_every": 50,
        "batch_metrics_window": 50,
        "batch_eval_pool": 4,

        # --- Logging and Saving Args ---
        "logging_steps": 100,
        "eval_strategy": "epoch",
        "eval_steps": 0,
        "save_strategy": "epoch",
        "save_steps": 0,
        "save_total_limit": 2,
        "early_stopping_patience": 5,
        "num_workers": 2,           
        "max_samples_training": 4500,
        "max_samples_validation": 500,
    }
