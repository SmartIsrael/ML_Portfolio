# **Rice Disease Classification Using Deep Learning**

This project implements a Convolutional Neural Network (CNN) for classifying rice diseases using transfer learning with the VGG16 architecture. The model identifies four types of rice diseases—*Bacterial Blight, Blast, Brown Spot, and False Smut.* The project includes data preprocessing, feature engineering, model training, evaluation, and bias mitigation to ensure fairness and accuracy.

---

## **Table of Contents**
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

## **Project Overview**
The goal of this project is to assist farmers in early disease detection by leveraging a deep learning model. Transfer learning with VGG16 was chosen for its robust feature extraction capabilities, reducing training time and improving performance on the rice disease classification task.

---

## **Dataset**
The dataset contains images of rice leaves affected by various diseases. Images are organized into subdirectories for each disease class and are processed into model-friendly formats. Data augmentation techniques such as rotation, flipping, and zooming were applied to improve generalization.

---

## **Data Preprocessing**
1. **Image Preprocessing**:
   - Images resized to 224x224 pixels.
   - Pixel values normalized to the range [0, 1].
   - Augmentation applied to enhance dataset diversity.

2. **Categorical Labels**:
   - Transformed into one-hot encoded vectors to align with the softmax activation.

3. **Bias Mitigation**:
   - Addressed class imbalances using SMOTE for oversampling.
   - Verified class distribution and performance metrics to ensure fairness.

---

## **Model Architecture**
The VGG16 model is used for feature extraction, with the top layers replaced by:
- A **Flatten** layer.
- A **Dense** layer (512 units, ReLU activation, L2 regularization).
- A **Dropout** layer (50% rate).
- An **Output** layer with softmax activation for four classes.

---

## **Evaluation**
The model was evaluated using:
- **Accuracy**: Achieved 94.12% on the test set.
- **Confusion Matrix**: Analyzed to measure per-class performance.
- **F1-Score**: Ensured robust evaluation of precision and recall.

---

## **Results**
- **Top 10 Features**: Identified using variance from VGG16 feature extraction.
- **Class Performance**: Consistent accuracy across all disease classes after bias mitigation.
- **Visualization**: t-SNE plots confirmed separability between disease classes.

---

## **Future Improvements**
- Expand dataset by incorporating diverse sources such as PlantVillage and field-collected data.
- Experiment with advanced architectures like ResNet or Inception for better accuracy.
- Integrate explainability techniques like Grad-CAM to visualize model decisions.

---

## **Data Storage**
Processed data is stored in a structured directory format locally, with backups on cloud storage (e.g., AWS S3) for scalability and security.

---

## **How to Run**
1. Clone this repository.
2. Install required libraries:
   ```bash
   pip install tensorflow opencv-python numpy matplotlib scikit-learn imblearn
   ```
3. Prepare the dataset in the `./Rice_Diseases` directory.
4. Run the training script to preprocess data, train the model, and evaluate results.

---

## **Contributors**
Developed as part of a portfolio project exploring the intersection of AI and agriculture. Contributions and suggestions are welcome!

---

This README provides a concise yet comprehensive overview of the project, focusing on critical aspects like preprocessing, architecture, and evaluation. Let me know if you'd like further refinements!
