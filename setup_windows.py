"""
Windows Setup Script for NeuroVandick
This script helps configure Windows for better memory management during AI training.
"""

import os
import subprocess
import sys
import psutil

def check_system_requirements():
    """Check if the system meets minimum requirements."""
    print("=== System Requirements Check ===")
    
    # Check RAM
    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"Total RAM: {ram_gb:.1f} GB")
    
    if ram_gb < 16:
        print("WARNING: Less than 16GB RAM detected. Consider upgrading for better performance.")
    
    # Check available disk space
    disk_usage = psutil.disk_usage('/')
    disk_gb = disk_usage.free / (1024**3)
    print(f"Available disk space: {disk_gb:.1f} GB")
    
    if disk_gb < 10:
        print("WARNING: Less than 10GB free disk space. Consider freeing up space.")
    
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.1f} GB")
            
            if gpu_memory < 6:
                print("WARNING: GPU memory is limited. The script will use memory optimizations.")
        else:
            print("CUDA not available. Training will be very slow on CPU.")
    except ImportError:
        print("PyTorch not installed. Please install it first.")

def set_environment_variables():
    """Set environment variables for better memory management."""
    print("\n=== Setting Environment Variables ===")
    
    env_vars = {
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_LAUNCH_BLOCKING": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"Set {key} = {value}")

def print_paging_file_instructions():
    """Print instructions for increasing the paging file size."""
    print("\n=== Paging File Configuration ===")
    print("To fix the 'paging file is too small' error, you need to increase your virtual memory:")
    print("\n1. Open System Properties (Win + Pause/Break)")
    print("2. Click 'Advanced system settings'")
    print("3. Under Performance, click 'Settings'")
    print("4. Click 'Advanced' tab")
    print("5. Under Virtual memory, click 'Change'")
    print("6. Uncheck 'Automatically manage paging file size'")
    print("7. Select your system drive (usually C:)")
    print("8. Choose 'Custom size'")
    print("9. Set Initial size to 16384 MB (16 GB)")
    print("10. Set Maximum size to 32768 MB (32 GB)")
    print("11. Click 'Set' then 'OK'")
    print("12. Restart your computer")
    print("\nThis requires administrator privileges and a system restart.")

def create_memory_optimized_script():
    """Create a memory-optimized version of the training script."""
    print("\n=== Creating Memory-Optimized Script ===")
    
    script_content = '''"""
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
    instruction_part = "<|start_header_id|>user<|end_header_id|>\\n\\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\\n\\n",
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
'''
    
    with open("src/main_memory_optimized.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("Created src/main_memory_optimized.py")

def main():
    """Main setup function."""
    print("=== NeuroVandick Windows Setup ===")
    
    check_system_requirements()
    set_environment_variables()
    print_paging_file_instructions()
    create_memory_optimized_script()
    
    print("\n=== Setup Complete ===")
    print("Next steps:")
    print("1. Increase your paging file size (see instructions above)")
    print("2. Restart your computer")
    print("3. Run: python src/main_memory_optimized.py")
    print("\nIf you still have issues, try:")
    print("- Close other applications to free up memory")
    print("- Use the memory-optimized script instead of main.py")

if __name__ == "__main__":
    main() 