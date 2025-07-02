# Deep Learning CNN Flower Classification

A comprehensive deep learning project that implements a Convolutional Neural Network (CNN) for multi-class flower image classification using TensorFlow and Keras.

## 🌸 Project Overview

This project demonstrates the complete pipeline of building, training, and evaluating a CNN model for classifying flower images into 5 categories:
- **Roses** (641 images)
- **Daisies** (633 images) 
- **Dandelions** (898 images)
- **Sunflowers** (699 images)
- **Tulips** (799 images)

**Total Dataset**: 3,670 flower images

## 🎯 Key Features

- **End-to-End CNN Implementation**: Complete deep learning pipeline from data loading to model evaluation
- **Data Preprocessing**: Image resizing, normalization, and train-test splitting
- **Overfitting Analysis**: Demonstrates overfitting detection and potential solutions
- **Multi-class Classification**: Handles 5 different flower categories
- **Comprehensive Evaluation**: Training and test accuracy analysis

## 🏗️ Technical Architecture

### Model Architecture
```
Sequential CNN Model:
├── Conv2D(16, 3x3) + ReLU + MaxPooling2D
├── Conv2D(32, 3x3) + ReLU + MaxPooling2D  
├── Conv2D(64, 3x3) + ReLU + MaxPooling2D
├── Flatten()
├── Dense(128) + ReLU
└── Dense(5) [Output Layer]
```

### Technical Specifications
- **Input Size**: 180x180x3 RGB images
- **Output Classes**: 5 flower categories
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Training Epochs**: 30
- **Data Split**: Train/Test split using scikit-learn

## 📊 Dataset Information

The project uses the TensorFlow Flowers dataset containing:
- **Source**: Google Storage (TensorFlow official dataset)
- **Format**: JPG images
- **Distribution**:
  - Roses: 641 images
  - Daisies: 633 images
  - Dandelions: 898 images
  - Sunflowers: 699 images
  - Tulips: 799 images

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7+
- pip package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/manoharpavuluri/dl-cnn-flowerclassification.git
   cd dl-cnn-flowerclassification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

4. **Open the notebook**
   - Navigate to `dl_cnn_flowerclassification.ipynb`
   - Run all cells sequentially

## 📋 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | ≥2.10.0 | Deep learning framework |
| opencv-python | ≥4.6.0 | Image processing |
| scikit-learn | ≥1.1.0 | Data splitting and utilities |
| matplotlib | ≥3.5.0 | Data visualization |
| numpy | ≥1.21.0 | Numerical computations |
| Pillow | ≥9.0.0 | Image handling |
| prettytable | ≥2.5.0 | Data display formatting |
| jupyter | ≥1.0.0 | Notebook environment |

## 🔧 Usage Instructions

### 1. Data Loading
The notebook automatically downloads the flower dataset from TensorFlow's official repository.

### 2. Data Preprocessing
- Images are resized to 180x180 pixels
- Pixel values are normalized to [0,1] range
- Data is split into training and test sets

### 3. Model Training
- CNN model is built using Keras Sequential API
- Training runs for 30 epochs
- Progress is monitored with accuracy metrics

### 4. Evaluation
- Model performance is evaluated on test set
- Overfitting analysis is performed

## 📈 Performance Results

### Training Performance
- **Final Training Accuracy**: ~100%
- **Training Loss**: Near zero
- **Training Time**: ~30 epochs

### Test Performance  
- **Test Accuracy**: ~65%
- **Test Loss**: ~2.79

### Overfitting Analysis
The model shows clear signs of overfitting:
- High training accuracy (100%) vs lower test accuracy (65%)
- This indicates the model memorizes training data rather than generalizing

## 🔍 Key Findings

1. **Overfitting Detection**: The model achieves 100% training accuracy but only 65% test accuracy
2. **Data Augmentation Need**: The notebook mentions data augmentation as a solution for overfitting
3. **Model Complexity**: The CNN architecture is appropriate for the task but needs regularization

## 🛠️ Technical Implementation Details

### Data Processing Pipeline
```python
# Image loading and preprocessing
for flower_name, images in flowers_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img, (180, 180))
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])
```

### Model Compilation
```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
```

## 🔮 Future Improvements

1. **Data Augmentation**: Implement rotation, zoom, and flip transformations
2. **Regularization**: Add dropout layers and L2 regularization
3. **Transfer Learning**: Use pre-trained models like VGG16 or ResNet
4. **Hyperparameter Tuning**: Optimize learning rate, batch size, and architecture
5. **Cross-validation**: Implement k-fold cross-validation for robust evaluation

## 📚 References

- **TensorFlow Tutorial**: [Image Classification Tutorial](https://www.tensorflow.org/tutorials/images/classification)
- **Dataset Source**: TensorFlow Flowers Dataset
- **Credits**: Based on TensorFlow official tutorial with modifications

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Performance improvements
- Additional features
- Documentation enhancements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Manohar Pavuluri**
- GitHub: [@manoharpavuluri](https://github.com/manoharpavuluri)

---

**Note**: This project serves as an educational resource for understanding CNN implementation, data preprocessing, and overfitting in deep learning models.

