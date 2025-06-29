import os
import torch
import gc
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt

# Set environment variables to manage memory and multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Clear GPU cache before starting
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

max_seq_length = 1024  # Reduced from 2048 to save memory
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",  # Switched to bnb-4bit model
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

print("Setting up PEFT model...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 8,  # Reduced from 16 to save memory
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 8,  # Reduced from 16
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

'''
data example:

<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hey there! How are you?<|eot_id|><|start_header_id|>user<|end_header_id|>

I'm great thanks!<|eot_id|>

'''

print("Setting up tokenizer...")
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

print("Loading and processing dataset...")
# Load a smaller subset of the dataset to reduce memory usage
dataset = load_dataset("mlabonne/FineTome-100k", split = "train[:1000]")  # Only first 1000 examples
dataset = standardize_sharegpt(dataset)
# Disable multiprocessing to avoid conflicts
dataset = dataset.map(formatting_prompts_func, batched = True)
print(f"Dataset size: {len(dataset)}")

# Verify dataset format
print("Sample conversation:")
print(dataset[5]["conversations"])
print("Sample text:")
print(dataset[5]["text"])

print("Setting up trainer...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 1,  # Disable multiprocessing
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 1,  # Reduced from 2
        gradient_accumulation_steps = 8,  # Increased from 4 to maintain effective batch size
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 30,  # Reduced from 60
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
        # Add memory optimization settings
        dataloader_pin_memory = False,
        remove_unused_columns = False,
        save_strategy = "no",  # Don't save checkpoints to save disk space
        evaluation_strategy = "no",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

print("Sample training data:")
print("Input IDs:")
print(tokenizer.decode(trainer.train_dataset[5]["input_ids"]))

space = tokenizer(" ", add_special_tokens = False).input_ids[0]
print("Labels:")
print(tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]]))

print("Starting training...")
try:
    trainer_stats = trainer.train()
    print("Training completed successfully!")
    print(f"Training stats: {trainer_stats}")
except Exception as e:
    print(f"Training failed with error: {e}")
    # Clean up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    raise e