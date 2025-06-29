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
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only, get_chat_template, standardize_sharegpt
from datasets import load_dataset

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

# Reduced parameters for memory efficiency
max_seq_length = 512  # Even smaller for memory
load_in_4bit = True

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
)

monitor_memory()

print("Setting up PEFT model...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 4,  # Minimal rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 4,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

monitor_memory()

print("Setting up tokenizer...")
tokenizer = get_chat_template(tokenizer, chat_template = "llama-3.1")

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

print("Loading dataset (small subset)...")
dataset = load_dataset("mlabonne/FineTome-100k", split = "train[:500]")  # Very small subset
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched = True)
print(f"Dataset size: {len(dataset)}")

monitor_memory()

print("Setting up trainer...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 1,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 16,  # Large accumulation to maintain effective batch size
        warmup_steps = 2,
        max_steps = 10,  # Very short training for testing
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
        dataloader_pin_memory = False,
        remove_unused_columns = False,
        save_strategy = "no",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

print("Starting training...")
try:
    trainer_stats = trainer.train()
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
