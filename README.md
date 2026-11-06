# Breast Cancer Classification Using Machine Learning

## a. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models to predict whether a breast tumor is benign or malignant.

Five different classification algorithms are trained using the Breast Cancer Wisconsin (Diagnostic) dataset. The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

The trained models are also integrated into a Streamlit application that allows users to upload test data, select a trained model, and view the corresponding evaluation results without retraining the model.

---

## b. Dataset Description

The Breast Cancer Wisconsin (Diagnostic) dataset is used for this classification problem.

The dataset contains 569 instances and 30 numerical predictive features. These features are computed from digitized images of breast mass samples and describe characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and related measurements.

The target variable is `diagnosis`, which represents whether the tumor is benign or malignant.

The target values are encoded as:

- **0 = Benign**
- **1 = Malignant**

Dataset Source: Breast Cancer Wisconsin (Diagnostic) Dataset, Kaggle

https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

---

## c. GitHub Repository Link

GitHub Repository:

**https://github.com/myantrapati/Breast-Cancer-Classification**

The repository contains the source code, Jupyter Notebook, Streamlit application, test dataset, requirements file, trained machine learning models, and preprocessing scaler required to execute the project.

---

## d. Models Used

The following five machine learning classification models were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Support Vector Machine (SVM)

The models were evaluated using the following metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Logistic Regression performed very well with an Accuracy of 0.9649 and achieved the highest AUC of 0.9960. The high F1 Score of 0.9512 and MCC of 0.9245 indicate strong and balanced classification performance. |
| Decision Tree | Decision Tree achieved an Accuracy of 0.9298 and an AUC of 0.9246. Although the model classified most samples correctly, it produced the lowest overall metric values among the five evaluated models. |
| KNN | KNN achieved an Accuracy of 0.9561 and AUC of 0.9823. It produced strong results after feature standardization, with an F1 Score of 0.9383 and MCC of 0.9058. |
| Naive Bayes | Naive Bayes achieved an Accuracy of 0.9386 and a high AUC of 0.9934. It achieved perfect Precision of 1.0000 but had a lower Recall of 0.8333, indicating that some malignant cases were not identified correctly. |
| Support Vector Machine | SVM achieved the highest Accuracy of 0.9737, highest F1 Score of 0.9630, and highest MCC of 0.9442. It also achieved perfect Precision of 1.0000, Recall of 0.9286, and a very high AUC of 0.9947. |

### Overall Winner for the Dataset

**Support Vector Machine (SVM) is the overall best-performing model for the Breast Cancer Wisconsin dataset.**

SVM achieved:

- **Accuracy:** 0.9737
- **AUC:** 0.9947
- **Precision:** 1.0000
- **Recall:** 0.9286
- **F1 Score:** 0.9630
- **MCC:** 0.9442

Although Logistic Regression achieved a slightly higher AUC of 0.9960, SVM achieved the highest Accuracy, F1 Score, and MCC while maintaining perfect Precision and strong Recall.

Therefore, considering the overall performance across all six evaluation metrics, **Support Vector Machine is selected as the winner for this dataset**.
