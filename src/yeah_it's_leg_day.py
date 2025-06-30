"""
Preprocessing Script
This script loads raw chat history files, formats them using a chat template,
and saves each formatted conversation as a single line in a new text file.
"""

import os
import torch
import gc
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import Dataset

def load_chatml_conversations(filepaths):
    """Load and parse chat history files in ChatML format into a list of conversations."""
    conversations = []
    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        # Split by <|begin_of_text|> if present
        for convo in text.split('<|begin_of_text|>'):
            convo = convo.strip()
            if not convo:
                continue
            # Split into turns
            turns = []
            segments = convo.split('<|eot_id|>')
            for seg in segments:
                seg = seg.strip()
                if not seg:
                    continue
                if seg.startswith('<|start_header_id|>user<|end_header_id|>'):
                    content = seg[len('<|start_header_id|>user<|end_header_id|>'):].strip()
                    turns.append({'role': 'user', 'content': content})
                elif seg.startswith('<|start_header_id|>assistant<|end_header_id|>'):
                    content = seg[len('<|start_header_id|>assistant<|end_header_id|>'):].strip()
                    turns.append({'role': 'assistant', 'content': content})
            # Only add if there are at least 2 turns (user+assistant)
            if len(turns) >= 2:
                conversations.append({'conversations': turns})
    return conversations

def main():
    print("Initializing tokenizer for preprocessing...")
    # Load a tiny model just to get the tokenizer
    model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
    _, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512, # Dummy value, not used for actual inference
        load_in_4bit=True,  # Dummy value, not used for actual inference
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    print("Loading raw chat history files...\n")
    chat_files = [
        'training_data/text/v1uz-tg-chat-history.txt',
        'training_data/text/group-tg-chat-history.txt',
        'training_data/text/viphase-tg-chat-history.txt',
    ]
    convo_data = load_chatml_conversations(chat_files)
    raw_dataset = Dataset.from_list(convo_data)

    # Define formatting function to apply chat template to each example
    def format_example_to_string(example):
        # example["conversations"] is a list of messages for a single example
        return tokenizer.apply_chat_template(
            example["conversations"],
            tokenize=False,
            add_generation_prompt=False
        )

    # Apply formatting and collect results as a list of strings
    # We explicitly convert to a Python list here to avoid Dataset/IterableDataset complexities
    formatted_strings = [format_example_to_string(item) for item in raw_dataset]

    output_file = "training_data/processed_chat_data.txt"
    print(f"Saving formatted chat data to {output_file}...\n")

    with open(output_file, 'w', encoding='utf-8') as f:
        for text_line in formatted_strings:
            f.write(text_line.strip() + "\n") # Ensure each conversation is on a single line

    print("Preprocessing complete!\n")

if __name__ == "__main__":
    # Set environment variables for Unsloth
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Clear memory (only if GPU is available, mostly for the FastLanguageModel dummy load)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    main()