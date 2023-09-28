# Breast Cancer Diagnosis with KNN and LDA

## Description:

Breast cancer is a major health concern, and early diagnosis is crucial for effective treatment. This project leverages machine learning techniques to develop a predictive model for breast cancer diagnosis using the Breast Cancer Wisconsin (Diagnostic) dataset. The goal is to create a tool that can assist medical professionals in distinguishing between malignant (cancerous) and benign (non-cancerous) breast tumors.

## Components:

Data Preprocessing:
The project starts by importing necessary Python libraries, including NumPy, Pandas, Seaborn, and Matplotlib, for data manipulation, visualization, and machine learning.
The dataset, containing various features extracted from breast cancer biopsies, is loaded into a Pandas DataFrame.
Data cleaning is performed, including the removal of unnecessary columns (e.g., 'id').

Data Exploration:
Descriptive statistics and visualizations are used to gain insights into the dataset.
Histograms are generated to visualize the distribution of features.

Data Preprocessing:
Data is divided into features (X) and the target variable (y).
A train-test split is performed to create training and testing datasets.
Feature scaling is applied to standardize the data using the StandardScaler.

Model Selection:
The project uses the k-Nearest Neighbors (KNN) algorithm as a machine learning model for breast cancer diagnosis.
Hyperparameter tuning is performed to find the optimal number of neighbors (k) using test accuracy as a metric.

Model Evaluation:
The KNN model's accuracy is evaluated on both the training and testing datasets.
A confusion matrix and classification report are generated to assess the model's performance.
A heatmap of the confusion matrix is plotted for visualization.

Dimensionality Reduction (Optional):
Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) are explored as dimensionality reduction techniques.
The KNN model is applied after dimensionality reduction to evaluate its impact on accuracy.

## Results:

The KNN model achieves a high accuracy of approximately 95.91% on the testing dataset, demonstrating its effectiveness in diagnosing breast cancer.
The model also shows strong performance on the training dataset, with an accuracy of approximately 98.24%.
Additionally, the impact of dimensionality reduction techniques like LDA is explored, resulting in an accuracy of approximately 97.08% on the testing dataset.
