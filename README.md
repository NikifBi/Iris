# Iris Flowers Classification

This project presents a complete machine learning workflow applied to the classic Iris dataset. The goal is to classify iris flowers into three species based on their physical characteristics.

The project includes:
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature selection
- Model training and hyperparameter tuning
- Performance evaluation and comparison

Several classification algorithms are implemented and compared, including both simple and advanced models.

## Dataset

The dataset used is the Iris dataset from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/53/iris

It contains:
- 150 samples
- 4 features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- 3 classes:
  - Setosa
  - Versicolor
  - Virginica

Two known incorrect entries in the dataset were corrected based on the original publication by R. A. Fisher.

## Methodology

The workflow follows standard machine learning practices:

1. Data preprocessing and validation
2. Exploratory Data Analysis (EDA)
3. Train-test split (stratified)
4. Feature selection using Chi-Squared test
5. Model training 
6. Hyperparameter tuning via cross-validation
7. Final evaluation on test data

Feature selection reduced the dataset to the two most informative features:
- Petal Length
- Petal Width

## Models

The following models were implemented and compared:

- DummyClassifier (baseline)
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVC)
- Logistic Regression
- Random Forest
- Multi-layer Perceptron (MLP)
- XGBoost

Hyperparameters were optimized using GridSearchCV and RandomizedSearchCV.

## Results

- All trained models significantly outperform the baseline DummyClassifier (~33% accuracy)
- Several models achieve near-perfect classification performance
- Cross-validation and test scores are closely aligned, indicating minimal overfitting
- Misclassifications occur only between Versicolor and Virginica
- Setosa is perfectly separable from the other classes

Despite similar performance, simpler models (e.g., Logistic Regression, KNN) are preferred due to interpretability and lower complexity.

## Limitations

- The dataset is small and highly structured
- Differences between models are minimal
- Results may not generalize to more complex, real-world datasets

## Installation

Clone the repository:

```bash
git clone https://github.com/NikifBi/Iris.git
cd iris-classification


