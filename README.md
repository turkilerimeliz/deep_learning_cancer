# 🧠 Histopathological Image Classification for Lung and Colon Cancer Using Deep Learning

This project applies deep learning to classify histopathological images of lung and colon cancer into five distinct tissue categories. Using the LC25000 dataset, convolutional neural networks are developed and evaluated to support the early and automated diagnosis of cancer subtypes through medical image analysis.

---

## 🩸 Background

Accurate and early diagnosis of lung and colon cancers is critical for treatment success. Manual inspection of histopathological slides, while effective, is time-consuming and prone to human error. Deep learning, particularly CNNs, offers the ability to automate this process with high accuracy. This project focuses on classifying histopathological images into five classes using both custom-built CNNs and transfer learning techniques.

---

## 🎯 Objectives

- Perform multi-class classification on histopathological images from lung and colon tissue.
- Compare performance of three models:
  - Baseline CNN
  - Enhanced CNN with RandomSearch Tuning (10 trials)
  - Transfer Learning with VGG16
- Analyze model sensitivity across different cancer types.

---

## 🛠️ Dataset and Methodology

- **Data**: LC25000 histopathological image dataset
- **Target**:
  - Colon benign tissue
  - Colon adenocarcinoma
  - Lung benign tissue
  - Lung squamous cell carcinoma
  - Lung adenocarcinoma
- **Sample**:
  - 25,000 images (augmented from original 1,250 images)
- **Preprocessing**:
  - Images resized from 768×768 to 120×120 pixels
  - One-hot encoding for target labels
  - No normalization (due to memory constraints)
- **Models**: Trained using stratified split;  60% training, 20% validation, 20% test.
- **Evaluation Metrics**: ⚖️ F1 Score, 📊 Recall, 🎯 Precision, 📈 ROC-AUC, ✅ Accuracy

---

## 🧪 Results

- **Baseline CNN**:
  - Images resized from 768×768 to 120×120 pixels
  - One-hot encoding for target labels
  - Validation/Test Accuracy: 72%
  - Overfitting after 9th epoch
  - Macro F1-Score (test): 0.71
  - Performance highly class-dependent
  - Strongest test performance: Lung benign tissue (F1 = 0.94, Recall = 0.93)
  - Weakest test performance: Lung adenocarcinoma (F1 = 0.51, Recall = 0.40)
    
   ### 🧱 Table 1: Architecture of the Baseline CNN Model
    | Layer                     | Parameters                                                                 |
    |--------------------------|----------------------------------------------------------------------------|
    | **Conv2D Layer 1**       | Filters: 128, Kernel size: 3x3, Activation: ReLU, Padding: same, Input shape: (120, 120, 3) |
    | **Max Pooling 1**        | Pool size: 2x2                                                             |
    | **Conv2D Layer 2**       | Filters: 64, Kernel size: 3x3, Activation: ReLU, Padding: same             |
    | **Max Pooling 2**        | Pool size: 2x2                                                             |
    | **Flatten**              | —                                                                          |
    | **Dense Layer 1**        | Units: 128, Activation: ReLU                                               |
    | **Dense Layer 2**        | Units: 32, Activation: ReLU                                                |
    | **Output Layer**         | Units: 5, Activation: Softmax                                              |
    | **Optimizer**            | Adam                                                                       |
    | **Loss Function**        | Categorical Crossentropy                                                   |
    | **Training Configuration** | Batch size: 32, Epochs: 10                                                |  


- **Enhanced CNN (RandomSearch Tuned)**:
   - Validation/Test Accuracy: 98%
   - Introduced additional Conv2D layer, dropout, early stopping
   - Best performance across all models
   - Macro F1-Score (test): 0.98
   - Notable Improvement: Lung adenocarcinoma F1 increased from 0.51 → 0.96
 
  ### 🧱 Table 2: Architecture of the Enhanced CNN Model
    | Layer                     | Parameters                                                                 |
    |--------------------------|----------------------------------------------------------------------------|
    | **Conv2D Layer 1**       | Filters: 128, Kernel size: 3x3, Activation: ReLU, Padding: same            |
    | **Max Pooling 1**        | Pool size: 3x3                                                             |
    | **Conv2D Layer 2**       | Filters: 256, Kernel size: 3x3, Activation: ReLU, Padding: same            |
    | **Max Pooling 2**        | Pool size: 3x3                                                             |
    | **Dropout 1**            | Rate: 0.2                                                                  |
    | **Conv2D Layer 3**       | Filters: 512, Kernel size: 5x5, Activation: ReLU, Padding: same            |
    | **Max Pooling 3**        | Pool size: 3x3                                                             |
    | **Dropout 2**            | Rate: 0.1                                                                  |
    | **Flatten**              | —                                                                          |
    | **Dense Layer 1**        | Units: 64                                                                  |
    | **Dense Layer 2**        | Units: 32                                                                  |
    | **Output Layer**         | Units: 5, Activation: Softmax                                              |
    | **Optimizer**            | Adam (learning rate = 1e-4)                                                |
    | **Loss Function**        | Categorical Crossentropy                                                   |
    | **Training Configuration** | Batch size: 64, Epochs: 32, Early stopping (patience = 5, monitor = val_loss) |

- **Transfer Learning (VGG16)**:
  - Validation/Test Accuracy: 97% / 96%
  - Faster convergence (peaked at epoch 3)
  - Good generalization but lower recall for lung adenocarcinoma (0.87)
  - Macro F1-Score (test): 0.96
---

## 📁 Repository Structure
📄 Histopathological Image Classification for Lung and Colon Cancer Using Deep Learning.pdf  
📓 histopathological_image_classification.ipynb  
📄 README.md  

---

## 🔍 Key Insights
- Enhanced CNN outperformed both baseline and transfer learning models.
- Class imbalance not an issue due to dataset balancing.
- Lung adenocarcinoma remains the most challenging class across all models.
- RandomSearch tuning significantly improved overall performance and class sensitivity.

## 🧠 Recommendations for Further Improvement
- Experiment with RMSprop optimizer (Masud et al., 2021; Masud et al., 2020; Mangal, Chaurasia & Khajanchi, 2020)
- Try connector convolution blocks
- Apply sharper image preprocessing (Masud et al., 2020)
- Consider batch minimization with stability control (Masud et al., 2021)

## ⚙️ Technologies Used

| **Library**                 | **Reference**                      |
|---------------------------|------------------------------------|
| **Pandas**                 | McKinney, 2010                     |
| **Numpy**                 | Harris et al., 2020                |
| **Matplotlib**             | Hunter, 2007                       |
| **Pillow (PIL)**          | Python Imaging Library             |
| **Scikit-learn**           | Pedregosa et al., 2011             |
| **TensorFlow / Keras**    | Abadi et al., 2016                 |
| **Keras VGG16**            | Simonyan & Zisserman, 2014         |
| **itertools / os**        | Python Standard Library            |

  
---
