# Table Detection Platform Guide

This guide covers how to run table detection training on Google Colab, Kaggle, and local environments.

## 🚀 Quick Start

```bash
# 1. Setup platform-specific environment
python setup_platform.py

# 2. Create sample dataset (for testing)
python create_sample_dataset.py

# 3. Convert data to YOLO format
python scripts/data_converter.py

# 4. Start training
python train_table_detection.py
```

## 📱 Platform Comparison

| Feature | Google Colab | Kaggle | Local |
|---------|-------------|--------|-------|
| **GPU Quota** | 12 hours | 30 hours | Unlimited |
| **Session Limit** | 12 hours | 9 hours | Unlimited |
| **Storage** | 79GB | 73GB temp + 20GB working | Unlimited |
| **Persistence** | Google Drive | Working dir only | Full control |
| **Best For** | Development & prototyping | Production training | Full control & large datasets |

---

## 🔵 Google Colab Setup

### Prerequisites
- Google account with Colab access
- Google Drive (recommended for persistence)

### Setup Steps

1. **Enable GPU Runtime**
   ```
   Runtime → Change runtime type → Hardware accelerator → GPU
   ```

2. **Mount Google Drive** (Optional but recommended)
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Clone/Upload Project**
   ```bash
   # Option 1: Clone from repository
   !git clone <your-repo-url> /content/table_detection

   # Option 2: Upload to Google Drive and access
   %cd /content/drive/MyDrive/table_detection
   ```

4. **Install Dependencies**
   ```bash
   !pip install -r requirements.txt
   ```

5. **Setup Environment**
   ```bash
   !python setup_platform.py
   ```

### Colab-Specific Features

- **Data Paths**: All data stored in `/content/data/`
- **Persistence**: Use Google Drive for long-term storage
- **GPU**: T4 GPU (16GB VRAM) typically available
- **Session Management**: Save work every 2-3 hours

### Storage Structure
```
/content/
├── data/                 # Training data
│   ├── raw/             # Original datasets
│   ├── processed/       # Preprocessed data
│   └── yolo/           # YOLO format data
├── checkpoints/        # Model checkpoints
├── logs/              # Training logs
└── results/           # Training results
```

### Colab Tips
- Enable GPU before starting training
- Use smaller batch sizes (8-16) to avoid memory issues
- Save checkpoints frequently
- Download models before session expires
- Use `%tensorboard --logdir logs` for monitoring

---

## 🟠 Kaggle Setup

### Prerequisites
- Kaggle account with phone verification
- GPU quota available (30 hours/week)

### Setup Steps

1. **Create New Notebook**
   - Go to Kaggle Notebooks
   - Create → Notebook → Accelerator: GPU

2. **Upload Project Files**
   ```bash
   # Option 1: Upload as Kaggle dataset
   # Upload your code as a private dataset, then add to notebook

   # Option 2: Direct file upload
   # Use Kaggle's file upload feature
   ```

3. **Setup Environment**
   ```bash
   !python ../input/table-detection-code/setup_platform.py
   ```

### Kaggle-Specific Features

- **Data Paths**: Large data in `/kaggle/tmp/data/` (73GB)
- **Working Dir**: Persistent storage in `/kaggle/working/` (20GB)
- **GPU**: P100 GPU (16GB VRAM) or T4
- **Datasets**: Access to public Kaggle datasets

### Storage Structure
```
/kaggle/
├── tmp/data/            # Large temporary data (73GB)
│   ├── raw/            # Original datasets
│   ├── processed/      # Preprocessed data
│   └── yolo/          # YOLO format data
├── working/            # Persistent storage (20GB)
│   ├── checkpoints/   # Model checkpoints
│   ├── logs/         # Training logs
│   ├── results/      # Training results
│   └── final_models/ # Models for download
├── input/             # Input datasets
└── output/           # Notebook output
```

### Kaggle Tips
- Use `/kaggle/tmp/` for large datasets
- Save important results in `/kaggle/working/`
- Upload trained models as Kaggle datasets for reuse
- Use Kaggle datasets for training data
- Take advantage of 30-hour GPU quota
- Save checkpoints every 5 epochs

---

## 💻 Local Development Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- At least 16GB RAM

### Setup Steps

1. **Clone Repository**
   ```bash
   git clone <your-repo-url>
   cd table_detection
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Environment**
   ```bash
   python setup_platform.py
   ```

### Local-Specific Features

- **Data Paths**: Relative paths (`./data/`)
- **Storage**: No limitations
- **GPU**: Full control over hardware
- **Time**: No session limits

### Storage Structure
```
./table_detection/
├── data/               # Training data
│   ├── raw/           # Original datasets
│   ├── processed/     # Preprocessed data
│   └── yolo/         # YOLO format data
├── checkpoints/       # Model checkpoints
├── logs/             # Training logs
├── results/          # Training results
└── scripts/          # Utility scripts
```

### Local Tips
- Use larger batch sizes for better performance
- Enable distributed training for multiple GPUs
- Use full datasets for best results
- Monitor GPU usage with `nvidia-smi`
- Consider using Docker for reproducibility

---

## ⚙️ Platform-Specific Configuration

The project automatically detects your platform and configures paths accordingly:

### Automatic Detection
```python
from scripts.platform_detector import get_platform_config

# Automatically detects platform and sets appropriate paths
config = get_platform_config()
print(f"Platform: {config.platform}")
print(f"Data root: {config.config['data_root']}")
```

### Manual Override
```python
# Force specific platform configuration
os.environ['FORCE_PLATFORM'] = 'kaggle'  # or 'colab' or 'local'
```

---

## 📊 Training Recommendations

### Google Colab
```python
# Recommended training parameters for Colab
train_params = {
    'epochs': 30,
    'batch': 8,
    'imgsz': 640,
    'patience': 10,
    'save_period': 5,
    'workers': 2,
    'cache': False,  # Disable caching to save memory
}
```

### Kaggle
```python
# Recommended training parameters for Kaggle
train_params = {
    'epochs': 100,
    'batch': 16,
    'imgsz': 640,
    'patience': 15,
    'save_period': 5,
    'workers': 2,
    'cache': False,  # Disable caching to save memory
}
```

### Local
```python
# Recommended training parameters for Local
train_params = {
    'epochs': 200,
    'batch': 32,
    'imgsz': 640,
    'patience': 20,
    'save_period': 10,
    'workers': 4,
    'cache': True,  # Enable caching if sufficient memory
}
```

---

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size
   - Disable caching
   - Use smaller image size

2. **Session Timeout (Colab/Kaggle)**
   - Save checkpoints frequently
   - Enable resume training
   - Download important models

3. **Slow Training**
   - Check GPU is enabled
   - Increase batch size if memory allows
   - Use mixed precision training

4. **Path Errors**
   - Run `python setup_platform.py` first
   - Check platform detection is correct
   - Verify directory structure

### Debug Commands
```bash
# Check platform detection
python -c "from scripts.platform_detector import get_platform_config; get_platform_config().print_platform_info()"

# Check GPU availability
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# Verify directory structure
ls -la /content/data/  # Colab
ls -la /kaggle/tmp/data/  # Kaggle
ls -la ./data/  # Local
```

---

## 📈 Performance Optimization

### Memory Optimization
- Use `cache=False` for limited memory
- Reduce batch size
- Use gradient checkpointing
- Clear cache regularly

### Speed Optimization
- Use mixed precision training (`amp=True`)
- Increase batch size for GPU utilization
- Use multiple workers for data loading
- Enable compile mode (PyTorch 2.0+)

### Storage Optimization
- Compress datasets when possible
- Use symlinks for large files
- Clean up temporary files regularly
- Save only essential checkpoints

---

## 🎯 Best Practices

1. **Always run platform setup first**
2. **Save work frequently on cloud platforms**
3. **Monitor GPU/memory usage**
4. **Use appropriate batch sizes for your platform**
5. **Test with sample data before using large datasets**
6. **Keep track of training progress**
7. **Download important models before session expires**

---

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify platform setup is correct
3. Check hardware requirements
4. Review training logs for errors
5. Use debug commands to diagnose issues