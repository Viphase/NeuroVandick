import os
from bs4 import BeautifulSoup

def format_turn(role, message_text):
    """Formats a single turn."""
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n{message_text}<|eot_id|>"

def process_html_file(file_path):
    """Processes a single HTML file and returns a list of formatted conversation strings."""
    print(f"Processing {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'lxml')
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Skipping.")
        return []

    conversations = []
    current_conversation_turns = []
    last_role = None
    assistant_name = "\u0412\u0430\u043d\u043e" # "Вано"

    for message_div in soup.find_all('div', class_='message'):
        if 'default' not in message_div.get('class', []):
            continue

        from_name_div = message_div.find('div', class_='from_name')
        sender = None
        if from_name_div:
            sender = from_name_div.get_text(strip=True).strip()

        text_div = message_div.find('div', class_='text')
        message_text = None
        if text_div:
            message_text = text_div.get_text(strip=True).replace("\n", " ").strip()
        
        if not message_text: # Skip empty messages
            continue

        role = "user"
        if sender == assistant_name:
            role = "assistant"
        
        # Heuristic for new conversation:
        # If it's the very first message, or if the role changes from assistant to user, or user to user.
        # This is a simplified heuristic and might not be perfect for all chat logs.
        if last_role is None or \
           (role == "user" and last_role == "assistant") or \
           (role == "user" and last_role == "user"):
            if current_conversation_turns: # Save previous conversation if exists
                conversations.append("<|begin_of_text|>" + "".join(current_conversation_turns))
            current_conversation_turns = [] # Start new conversation
        
        current_conversation_turns.append(format_turn(role, message_text))
        last_role = role
    
    # Add the last conversation
    if current_conversation_turns:
        conversations.append("<|begin_of_text|>" + "".join(current_conversation_turns))

    print(f"Found {len(conversations)} conversations in {file_path}")
    return conversations

def main():
    """Main function to process all HTML files and save the output."""
    input_files = ['messages.html', 'messages2.html', 'messages3.html', 'messages4.html']
    all_formatted_conversations = []

    for file_name in input_files:
        all_formatted_conversations.extend(process_html_file(file_name))

    output_file = 'training_data/text/html_chat_history.txt'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for convo in all_formatted_conversations:
            f.write(convo + "\n") # Each conversation on a new line

    print(f"\nSuccessfully processed all HTML files. Output saved to {output_file}")
    print(f"Total conversations processed from HTML: {len(all_formatted_conversations)}")

if __name__ == "__main__":
    main() 