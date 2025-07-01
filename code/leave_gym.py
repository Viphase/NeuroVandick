import os
from unsloth import FastLanguageModel

def main():
    # Path to your trained model directory
    trained_model_dir = os.path.join("./outputs")  # or provide absolute path if needed

    # The GGUF file will be saved here
    gguf_output_path = "./models/your_model"

    # Use the same max_seq_length as in training
    max_seq_length = 512

    print(f"Loading model from {trained_model_dir} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = trained_model_dir,
        max_seq_length = max_seq_length,
        dtype = None,         # Auto-detect
        load_in_4bit = True,  # Or False, as you prefer
    )

    print(f"Exporting model to GGUF format at {gguf_output_path} ...")
    model.save_pretrained_gguf(gguf_output_path, tokenizer=tokenizer)
    print("Export complete!")

if __name__ == "__main__":
    main()