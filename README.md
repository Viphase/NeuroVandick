# NeuroVandick

A fine-tuning framework for language models using Unsloth, optimized for Windows systems with limited GPU memory.

## 🚀 Quick Start (Windows)

### Option 1: Automated Setup (Recommended)
```bash
# Run the automated setup
setup_windows.bat
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run Windows setup script
python setup_windows.py

# Run the memory-optimized training script
python src/main_memory_optimized.py
```

## 🔧 System Requirements

- **OS**: Windows 10/11
- **RAM**: 16GB+ (8GB minimum with optimizations)
- **GPU**: NVIDIA GPU with 6GB+ VRAM (RTX 3070 or better)
- **Storage**: 10GB+ free space
- **Python**: 3.8+

## 📋 Prerequisites

### 1. Increase Virtual Memory (Paging File)
**CRITICAL**: You must increase your Windows paging file size to avoid "paging file is too small" errors.

1. Press `Win + Pause/Break` to open System Properties
2. Click "Advanced system settings"
3. Under Performance, click "Settings"
4. Click "Advanced" tab
5. Under Virtual memory, click "Change"
6. Uncheck "Automatically manage paging file size"
7. Select your system drive (usually C:)
8. Choose "Custom size"
9. Set Initial size to **16384 MB** (16 GB)
10. Set Maximum size to **32768 MB** (32 GB)
11. Click "Set" then "OK"
12. **Restart your computer**

### 2. Close Other Applications
- Close browsers, games, and other memory-intensive applications
- Ensure at least 8GB RAM is available

## 🎯 Training Scripts

### Main Script (`src/main.py`)
- Standard training configuration
- Uses 1B model with moderate optimizations
- Good for systems with 8GB+ GPU memory

### Memory-Optimized Script (`src/main_memory_optimized.py`)
- Aggressive memory optimizations
- Uses smaller model parameters
- Reduced dataset size
- Real-time memory monitoring
- **Recommended for RTX 3070 (8GB VRAM)**

## 🛠️ Troubleshooting

### CUDA Out of Memory
**Symptoms**: `RuntimeError: CUDA error: out of memory`

**Solutions**:
1. Use the memory-optimized script: `python src/main_memory_optimized.py`
2. Close other GPU applications (games, browsers)
3. Reduce `max_seq_length` in the script
4. Reduce `per_device_train_batch_size` to 1
5. Increase `gradient_accumulation_steps`

### Paging File Too Small
**Symptoms**: `OSError: The paging file is too small for this operation to complete. (os error 1455)`

**Solutions**:
1. **Follow the paging file setup instructions above**
2. Restart your computer after changing paging file
3. Ensure you have enough free disk space

### Multiprocessing Conflicts
**Symptoms**: Multiple error traces, repeated Unsloth messages

**Solutions**:
1. The scripts now disable multiprocessing automatically
2. Set `dataset_num_proc = 1` in training arguments
3. Set `TOKENIZERS_PARALLELISM = "false"` environment variable

### Model Loading Issues
**Symptoms**: `ValueError: Some modules are dispatched on the CPU or the disk`

**Solutions**:
1. Use the 1B model instead of 3B: `"unsloth/Llama-3.2-1B-Instruct"`
2. Enable 4-bit quantization: `load_in_4bit = True`
3. Reduce LoRA rank: `r = 4` or `r = 8`

## 📊 Memory Optimization Features

### Environment Variables
```python
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
```

### Model Optimizations
- 4-bit quantization (`load_in_4bit = True`)
- Gradient checkpointing (`use_gradient_checkpointing = "unsloth"`)
- Reduced LoRA rank (`r = 4` or `r = 8`)
- Smaller sequence length (`max_seq_length = 512`)

### Training Optimizations
- Batch size of 1 with gradient accumulation
- Disabled multiprocessing
- Memory monitoring
- Automatic cleanup on errors

## 🔍 Monitoring

The memory-optimized script includes real-time monitoring:
- GPU memory usage (allocated and reserved)
- RAM usage percentage
- Automatic memory cleanup on errors

## 📁 Project Structure

```
NeuroVandick/
├── src/
│   ├── main.py                    # Standard training script
│   └── main_memory_optimized.py   # Memory-optimized training script
├── setup_windows.py               # Windows setup and diagnostics
├── setup_windows.bat              # Automated setup batch file
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🎯 Performance Tips

1. **Use the memory-optimized script** for RTX 3070 or similar 8GB cards
2. **Increase paging file** before training
3. **Close other applications** to free up memory
4. **Monitor memory usage** during training
5. **Start with small datasets** to test your setup

## 🆘 Getting Help

If you encounter issues:

1. Run `python setup_windows.py` for system diagnostics
2. Check the troubleshooting section above
3. Ensure you've increased the paging file size
4. Try the memory-optimized script
5. Monitor your system resources during training

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
