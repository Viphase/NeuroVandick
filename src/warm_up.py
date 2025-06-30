import os
from bs4 import BeautifulSoup

def format_message(role, message):
    """Formats a message into the required string format."""
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n{message}<|eot_id|>"

def process_html_file(file_path):
    """Processes a single HTML file and returns a list of formatted messages."""
    print(f"Processing {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'lxml')
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Skipping.")
        return []

    messages = []
    last_from = None
    
    # Assistant's name. Using Unicode escape sequence for robustness.
    assistant_name = "\u0412\u0430\u043d\u043e"  # "Вано"

    for message_div in soup.find_all('div', class_='message'):
        if 'default' not in message_div.get('class', []):
            continue

        from_name_div = message_div.find('div', class_='from_name')
        if from_name_div:
            last_from = from_name_div.get_text(strip=True)

        text_div = message_div.find('div', class_='text')
        if text_div:
            message_text = text_div.get_text(strip=True).replace("\n", " ").strip()
            if not message_text:
                continue

            role = "user"
            if last_from:
                sender = last_from.strip()
                if sender == assistant_name:
                    role = "assistant"
            
            messages.append(format_message(role, message_text))

    print(f"Found {len(messages)} messages in {file_path}")
    return messages

def main():
    """Main function to process all HTML files and save the output."""
    input_files = ['messages.html', 'messages2.html', 'messages3.html', 'messages4.html']
    all_formatted_messages = ["<|begin_of_text|>"]

    for file_name in input_files:
        all_formatted_messages.extend(process_html_file(file_name))

    output_content = "".join(all_formatted_messages)

    output_file = 'training_data/prepared_data.txt'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_content)

    print(f"\nSuccessfully processed all files. Output saved to {output_file}")
    print(f"Total messages processed: {len(all_formatted_messages) - 1}")


if __name__ == "__main__":
    main() 