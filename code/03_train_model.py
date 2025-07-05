"""
code/workout.py - Main training script (the heavy lifting!)
This is where the actual model training happens
"""

import os
import torch
import gc
import psutil
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
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

print("🏋️ Starting Workout (Model Training)")
print("=====================================")
monitor_memory()

if __name__ == '__main__':
    # Training parameters
    max_seq_length = 512
    load_in_4bit = True

    print("\n💪 Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
    )

    monitor_memory()

    print("\n🎯 Setting up PEFT (adding LoRA weights)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 4,  # LoRA rank
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 4,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = True,
        random_state = 3407,
    )

    monitor_memory()

    print("\n📝 Setting up tokenizer...")
    tokenizer = get_chat_template(tokenizer, chat_template = "llama-3.1")

    # Load the preprocessed dataset
    print("\n📚 Loading training data...")
    dataset = load_dataset("text", data_files="training_data/text/processed_chat_data.txt", split="train")
    print(f"Dataset size: {len(dataset) if hasattr(dataset, '__len__') else 'Unknown'}")

    monitor_memory()

    print("\n🏃 Setting up trainer...")
    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        dataset_text_field = "text",
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 16,
            warmup_steps = 5,
            max_steps = 100,  # Increase for better training
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none",
            save_strategy = "steps",
            save_steps = 50,
        ),
    )

    # Train on responses only (not on user messages)
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    print("\n🔥 Starting training...")
    print("This will take a while. Time for a real workout! 💪")
    
    try:
        trainer_stats = trainer.train()
        
        # Save the model
        print("\n💾 Saving trained model...")
        model.save_pretrained("outputs")
        tokenizer.save_pretrained("outputs")
        
        print("\n✅ Training completed successfully!")
        print(f"Training stats: {trainer_stats}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise e
    finally:
        monitor_memory()
        print("\n🏁 Workout complete! Time to leave the gym (run leave_gym.py)")

