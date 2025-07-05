import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

def check_and_process_voice_samples():
    """Check existing voice samples and process them for Coqui TTS"""
    
    # Your current voice files location
    current_voice_dir = "training_data/voice/processed"
    
    print("🎤 Voice Warm-Up: Checking Vandick's voice samples")
    print("=" * 50)
    
    # List existing files
    voice_files = list(Path(current_voice_dir).glob("*.wav"))
    
    if not voice_files:
        print("❌ No voice samples found!")
        return
    
    print(f"\n✅ Found {len(voice_files)} voice samples:")
    
    for i, file in enumerate(voice_files, 1):
        print(f"\n📁 Sample {i}: {file.name}")
        
        try:
            # Load and analyze
            audio, sr = librosa.load(file, sr=None)
            duration = len(audio) / sr
            
            print(f"   Duration: {duration:.1f} seconds")
            print(f"   Sample rate: {sr} Hz")
            
            # Check if resampling needed
            if sr != 22050:
                print(f"   ⚠️ Resampling from {sr}Hz to 22050Hz...")
                audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
                sr = 22050
                
                # Save resampled version
                output_path = file.parent / f"{file.stem}_22k.wav"
                sf.write(output_path, audio, sr)
                print(f"   ✅ Saved resampled: {output_path.name}")
            else:
                print(f"   ✅ Already at correct sample rate")
                
            # Check duration
            if duration < 10:
                print(f"   ⚠️ Short sample - consider using longer clips")
            elif duration > 30:
                print(f"   ⚠️ Long sample - consider splitting")
            else:
                print(f"   ✅ Good duration for voice cloning")
                
        except Exception as e:
            print(f"   ❌ Error processing: {e}")
    
    print("\n💡 Tips for best results:")
    print("   - Use 10-30 second clips")
    print("   - Ensure only Vandick is speaking")
    print("   - Minimize background noise")
    print("   - Have 3-5 different samples for variety")

if __name__ == "__main__":
    check_and_process_voice_samples()

