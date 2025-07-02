# Setup Guide - Deep Learning CNN Flower Classification

This guide provides step-by-step instructions for setting up and running the flower classification project.

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.14+, or Ubuntu 18.04+
- **Python**: 3.7 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 2GB free space
- **GPU**: Optional but recommended for faster training

### Recommended Requirements
- **RAM**: 16GB or more
- **GPU**: NVIDIA GPU with CUDA support
- **Storage**: SSD with 5GB free space

## 🐍 Python Environment Setup

### Option 1: Using Conda (Recommended)

1. **Install Anaconda/Miniconda**
   ```bash
   # Download from https://docs.conda.io/en/latest/miniconda.html
   # Or install Anaconda from https://www.anaconda.com/products/distribution
   ```

2. **Create a new conda environment**
   ```bash
   conda create -n flower-classification python=3.9
   conda activate flower-classification
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using Virtual Environment

1. **Create virtual environment**
   ```bash
   python -m venv flower-env
   
   # On Windows
   flower-env\Scripts\activate
   
   # On macOS/Linux
   source flower-env/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🔧 Installation Steps

### Step 1: Clone the Repository
```bash
git clone https://github.com/manoharpavuluri/dl-cnn-flowerclassification.git
cd dl-cnn-flowerclassification
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

## 🚀 Running the Project

### Method 1: Jupyter Notebook (Recommended)

1. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```

2. **Open the notebook**
   - Navigate to `dl_cnn_flowerclassification.ipynb`
   - Click to open

3. **Run the cells**
   - Use `Shift + Enter` to run individual cells
   - Or use `Cell > Run All` to run the entire notebook

### Method 2: Google Colab

1. **Open in Colab**
   - Click the "Open in Colab" button in the notebook
   - Or upload the notebook to Google Colab

2. **Install dependencies in Colab**
   ```python
   !pip install opencv-python prettytable
   ```

3. **Run the notebook**

## 📊 Expected Outputs

### Data Loading
- Dataset will be automatically downloaded (~230MB)
- You should see download progress in the notebook

### Training Progress
```
Epoch 1/30
86/86 [==============================] - 8s 22ms/step - loss: 1.3777 - accuracy: 0.4182
...
Epoch 30/30
86/86 [==============================] - 2s 20ms/step - loss: 1.7771e-04 - accuracy: 1.0000
```

### Final Results
```
29/29 [==============================] - 1s 22ms/step - loss: 2.7883 - accuracy: 0.6492
```

## 🐛 Troubleshooting

### Common Issues

#### 1. TensorFlow Installation Issues
```bash
# If you get TensorFlow installation errors
pip uninstall tensorflow
pip install tensorflow==2.10.0
```

#### 2. OpenCV Issues
```bash
# If OpenCV fails to install
pip uninstall opencv-python
pip install opencv-python-headless
```

#### 3. Memory Issues
- Reduce batch size in the model
- Close other applications
- Use Google Colab for cloud computing

#### 4. CUDA/GPU Issues
```bash
# Check if TensorFlow sees your GPU
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

### Performance Optimization

#### For Faster Training
1. **Use GPU**: Install CUDA and cuDNN for NVIDIA GPUs
2. **Reduce image size**: Change from 180x180 to 128x128
3. **Use data augmentation**: Implement ImageDataGenerator

#### For Memory Optimization
1. **Reduce batch size**: Modify the model.fit() call
2. **Use mixed precision**: Enable TensorFlow mixed precision
3. **Gradient accumulation**: Implement gradient accumulation

## 📁 Project Structure

```
dl-cnn-flowerclassification/
├── dl_cnn_flowerclassification.ipynb  # Main notebook
├── requirements.txt                   # Dependencies
├── README.md                         # Project documentation
├── SETUP.md                          # This setup guide
├── LICENSE                           # License file
└── datasets/                         # Downloaded dataset (auto-created)
    └── flower_photos/
        ├── roses/
        ├── daisy/
        ├── dandelion/
        ├── sunflowers/
        └── tulips/
```

## 🔍 Verification Checklist

- [ ] Python 3.7+ installed
- [ ] All dependencies installed successfully
- [ ] Jupyter notebook launches without errors
- [ ] Dataset downloads automatically
- [ ] Model training starts without errors
- [ ] Training completes with expected accuracy
- [ ] Test evaluation runs successfully

## 📞 Support

If you encounter any issues:

1. **Check the troubleshooting section above**
2. **Search existing GitHub issues**
3. **Create a new issue with:**
   - Your operating system
   - Python version
   - Error message
   - Steps to reproduce

## 🎯 Next Steps

After successful setup:

1. **Experiment with hyperparameters**
2. **Try data augmentation techniques**
3. **Implement transfer learning**
4. **Add model evaluation metrics**
5. **Deploy the model**

---

**Happy Coding! 🌸** 