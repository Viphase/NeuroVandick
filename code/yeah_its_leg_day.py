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

def load_and_parse_chat_files(filepaths, tokenizer):
    """Load and parse various chat history files, formatting them consistently."""
    all_formatted_conversations = []

    for filepath in filepaths:
        print(f"Processing {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        
        # Assume each line is a separate conversation if it's already pre-formatted
        # or split by <|begin_of_text|> if it's the raw Telegram export
        if "<|begin_of_text|>" in raw_content or filepath.endswith("html_chat_history.txt"):
            conversations_raw = raw_content.split("<|begin_of_text|>")
            for convo_text in conversations_raw:
                convo_text = convo_text.strip()
                if convo_text:
                    # For already formatted data, we just append it as is.
                    # We trust warm_up.py for HTML data and previous runs for Telegram data.
                    all_formatted_conversations.append("<|begin_of_text|>" + convo_text)
        else:
            # For raw Telegram files, parse by speaker name and filter empty messages
            current_conversation_turns = []
            last_role = None
            assistant_name = "Вано"

            lines = raw_content.splitlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                role = None
                message_text = ""
                speaker_name = None

                # Attempt to parse a common chat log format: [TIMESTAMP] Name: Message
                # Using a simple split, more robust regex might be needed for complex logs
                parts = line.split(': ', 1) # Split only on the first ': ' to get name and rest of message
                if len(parts) == 2:
                    potential_speaker_time = parts[0]
                    # Try to extract name, assuming it's after a timestamp in square brackets
                    if ']' in potential_speaker_time:
                        name_part = potential_speaker_time.split(']')[-1].strip()
                        if name_part:
                            speaker_name = name_part
                            message_text = parts[1].strip()
                    else: # If no timestamp, assume the first part is the speaker name
                        speaker_name = potential_speaker_time.strip()
                        message_text = parts[1].strip()
                else:
                    # If it doesn't match the "Name: Message" pattern,
                    # assume it's a continuation of the last message's content
                    if current_conversation_turns:
                        current_conversation_turns[-1]['content'] += "\n" + line
                    continue # Don't process as a new turn if no clear speaker

                if speaker_name:
                    if speaker_name.lower() == assistant_name.lower():
                        role = 'assistant'
                    elif speaker_name.lower() in ["system", "бот"]:
                        role = speaker_name.lower()
                    else:
                        role = 'user' # All other names are users

                if not role or not message_text: # Skip if role not determined or message is empty
                    continue

                # Heuristic for new conversation:
                # Start a new conversation segment if:
                # 1. It's the very first message processed (last_role is None).
                # 2. A 'user' speaks after an 'assistant'. (Normal conversational flow)
                # 3. A 'user' speaks after another 'user'. (Indicates assistant didn't respond, new convo starts)
                # 4. A 'system' message appears. (System message usually starts new context)
                # 5. An 'assistant' speaks after another 'assistant'. (Indicates new assistant-led convo or error in parsing)

                start_new_segment = False
                if last_role is None:
                    start_new_segment = True
                elif role == "user" and last_role == "assistant":
                    start_new_segment = True
                elif role == "user" and last_role == "user":
                    start_new_segment = True
                elif role == "system":
                    start_new_segment = True
                elif role == "assistant" and last_role == "assistant":
                    start_new_segment = True

                if start_new_segment and current_conversation_turns:
                    # Before starting a new segment, process the existing one
                    try:
                        formatted_text = tokenizer.apply_chat_template(
                            current_conversation_turns,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                        all_formatted_conversations.append("<|begin_of_text|>" + formatted_text)
                    except Exception as e:
                        print(f"Skipping malformed conversation: {e} - {current_conversation_turns}")
                    current_conversation_turns = [] # Reset for new segment

                current_conversation_turns.append({'role': role, 'content': message_text})
                last_role = role

            # Add the last conversation segment after the loop finishes
            if current_conversation_turns:
                try:
                    formatted_text = tokenizer.apply_chat_template(
                        current_conversation_turns,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    all_formatted_conversations.append("<|begin_of_text|>" + formatted_text)
                except Exception as e:
                    print(f"Skipping malformed final conversation: {e} - {current_conversation_turns}")

    return all_formatted_conversations

def main():
    print("Initializing tokenizer for preprocessing...")
    # Load a tiny model just to get the tokenizer for Llama 3.2
    model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    _, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512, # Dummy value
        load_in_4bit=True,
    )
    # Ensure we get the correct chat template for Llama 3.2
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.2")

    print("Gathering raw chat history files...")
    input_dir = 'training_data/text/'
    # Get all .txt files in the directory, excluding the output file
    all_raw_chat_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith('.txt') and f != 'processed_chat_data.txt' and f != 'example.txt'
    ]

    # Process and format all raw chat files
    print(f"Found {len(all_raw_chat_files)} raw chat files to process.")
    all_processed_conversations = load_and_parse_chat_files(all_raw_chat_files, tokenizer)

    output_file = "training_data/text/processed_chat_data.txt"
    print(f"Saving all formatted chat data to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for text_line in all_processed_conversations:
            f.write(text_line.strip() + "\n")

    print("Preprocessing complete!\n")
    print(f"Total conversations saved to processed_chat_data.txt: {len(all_processed_conversations)}")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    main()