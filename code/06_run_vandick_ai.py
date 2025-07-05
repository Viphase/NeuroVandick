"""
code/vandick_speaks.py - Complete Vandick AI with voice
This combines your trained text model with voice synthesis
"""

import os
import torch
from pathlib import Path
from TTS.api import TTS
from unsloth import FastLanguageModel
import pygame
import time

class VandickAI:
    def __init__(self):
        print("🚀 Initializing Vandick AI...")
        
        # Check device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Using device: {self.device}")
        
        # Load text model
        print("🧠 Loading Vandick's brain (text model)...")
        self.load_text_model()
        
        # Load voice model
        print("🎤 Loading Vandick's voice...")
        self.load_voice_model()
        
        # Initialize audio player
        pygame.mixer.init()
        
        print("✅ Vandick AI ready!\n")
    
    def load_text_model(self):
        """Load the fine-tuned text model"""
        model_path = "outputs"  # Your trained model
        
        if not os.path.exists(model_path):
            print("   ⚠️ No trained model found. Using base model.")
            model_path = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )
    
    def load_voice_model(self):
        """Load Coqui TTS with Vandick's voice"""
        # Find voice samples
        voice_dir = Path("training_data/voice/processed")
        self.voice_samples = list(voice_dir.glob("*.wav"))
        
        if not self.voice_samples:
            print("   ⚠️ No voice samples found! Voice will be disabled.")
            self.tts = None
            return
        
        print(f"   Found {len(self.voice_samples)} voice samples")
        
        # Load TTS model
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
    
    def generate_text(self, prompt):
        """Generate text response"""
        # Format for chat
        formatted = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # Tokenize
        inputs = self.tokenizer(formatted, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response
        if "assistant<|end_header_id|>" in response:
            response = response.split("assistant<|end_header_id|>")[-1].strip()
            response = response.split("<|eot_id|>")[0].strip()
        
        return response
    
    def speak(self, text, language="ru"):
        """Convert text to speech with Vandick's voice"""
        if not self.tts or not self.voice_samples:
            print("🔇 [Voice disabled - no samples]")
            return None
        
        # Use first 3 samples for best quality
        samples = [str(s) for s in self.voice_samples[:3]]
        
        # Generate speech
        output_path = "temp_speech.wav"
        self.tts.tts_to_file(
            text=text,
            speaker_wav=samples,
            language=language,
            file_path=output_path
        )
        
        # Play audio
        pygame.mixer.music.load(output_path)
        pygame.mixer.music.play()
        
        return output_path
    
    def chat(self, message):
        """Complete chat interaction"""
        print(f"\n👤 You: {message}")
        
        # Generate response
        response = self.generate_text(message)
        print(f"🤖 Vandick: {response}")
        
        # Speak if voice available
        if self.tts:
            # Auto-detect language
            lang = "ru" if any('\u0400' <= c <= '\u04FF' for c in response) else "en"
            self.speak(response, lang)
            
            # Wait for speech to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        
        return response


def main():
    print("🎮 Vandick AI - Text + Voice")
    print("============================")
    
    # Initialize
    vandick = VandickAI()
    
    # Interactive chat
    print("\nType 'exit' to quit")
    print("-" * 30)
    
    while True:
        try:
            user_input = input("\n👤 You: ")
            
            if user_input.lower() in ['exit', 'quit', 'выход']:
                print("\n👋 До свидания!")
                break
            
            if user_input.strip():
                vandick.chat(user_input)
                
        except KeyboardInterrupt:
            print("\n\n👋 Прервано. До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()