"""
Memory-Optimized Training Script
This version includes additional memory management features.
"""

import os
import torch
import gc
import psutil
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only, get_chat_template
from datasets import load_dataset, Dataset, IterableDataset

def monitor_memory():
    """Monitor and print memory usage."""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"GPU Memory: {gpu_memory:.2f} GB allocated, {gpu_memory_reserved:.2f} GB reserved")
    
    ram_usage = psutil.virtual_memory()
    print(f"RAM: {ram_usage.percent}% used ({ram_usage.used / (1024**3):.1f} GB / {ram_usage.total / (1024**3):.1f} GB)")

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Clear memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

print("=== Memory-Optimized Training Started ===")
monitor_memory()

if __name__ == '__main__':
    max_seq_length = 1024 
    load_in_4bit = True

    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Llama-3.2-1B-bnb-4bit",
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
    )

    monitor_memory()

    print("Setting up PEFT model...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 4,  # Minimal rank
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Added MLP layers for better patching
        lora_alpha = 4,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = True,
        random_state = 3407,
    )

    monitor_memory()

    print("Setting up tokenizer...")
    tokenizer = get_chat_template(tokenizer, chat_template = "llama-3.2")

    # Load the preprocessed dataset
    print("Loading preprocessed chat data...\n")
    # Load as a text dataset, where each line is an example
    dataset = load_dataset("text", data_files="training_data/text/processed_chat_data.txt", split="train")

    # Only print length if possible and not IterableDataset
    if (isinstance(dataset, (list, Dataset)) or hasattr(dataset, "__len__")) and not isinstance(dataset, IterableDataset):
        print(f"Dataset size: {len(dataset)}\n")
    else:
        print("Dataset is not sized (IterableDataset).\n")

    monitor_memory()

    print("Setting up trainer...")
    # The dataset now directly contains the 'text' field
    if isinstance(dataset, (Dataset, IterableDataset)):
        trainer = SFTTrainer(
            model = model,
            train_dataset = dataset,
            dataset_text_field = "text", # Point to the 'text' column created by load_dataset("text", ...)
            args = TrainingArguments(
                per_device_train_batch_size = 4,
                gradient_accumulation_steps = 8,  # Large accumulation to maintain effective batch size
                warmup_steps = 5,
                max_steps = 200,
                learning_rate = 2e-4,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = "outputs/",
                report_to = "none",
                dataloader_pin_memory = False,
                remove_unused_columns = False,
                save_strategy = "no",
            ),
            # No formatting_func needed here, as data is already preprocessed
        )
    else:
        raise ValueError("train_dataset must be a Dataset or IterableDataset, not {}".format(type(dataset)))

    print("Starting training...")
    try:
        trainer_stats = trainer.train()
        # After training
        model.save_pretrained("outputs/")
        tokenizer.save_pretrained("outputs/")
        print("Training completed successfully!")
        print(f"Training stats: {trainer_stats}")
    except Exception as e:
        print(f"Training failed: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise e
    finally:
        monitor_memory()