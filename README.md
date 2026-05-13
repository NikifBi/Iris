# Iris Flowers Classification

A Python project demonstrating a machine learning workflow on the classical Iris dataset. This repository includes data loading, model training, hyperparameter tuning, and visualization code for six classifiers.

## Project Overview

The goal is to classify iris flower samples into three species using physical measurements.

Key features:
- Custom data loader for the Iris dataset
- Train/test split with stratified sampling
- Pipeline-based model tuning for robust preprocessing
- Hyperparameter optimization using `GridSearchCV` and `RandomizedSearchCV`
- Decision boundary and confusion matrix visualization support

## Repository Structure

- `data/`
  - `iris.data` — Iris dataset CSV file
  - `data_info.txt` — Dataset metadata and reference information
- `notebook/`
  - `iris_jup.ipynb` — Jupyter notebook for exploration and analysis
- `src/`
  - `data_loader_iris.py` — CSV loader and class label mapping
  - `iris_dataset.py` — classifier pipelines, tuning routines, and plotting helpers
- `requirements.txt` — dependencies for the project

> Note: `iris_dataset.py` contains reusable training and visualization utilities derived from the notebook workflow.

## Installation

From the repository root:

```bash
python -m pip install -r requirements.txt
```

If you use a virtual environment, activate it before installing.

## Usage

Open the Jupyter notebook:

```bash
jupyter notebook notebook/iris_jup.ipynb

```

## Dataset

This project uses the Iris dataset from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/53/iris

Dataset details:
- 150 samples
- 4 input features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- 3 target classes:
  - Iris-setosa
  - Iris-versicolor
  - Iris-virginica

The dataset is stored locally in `data/iris.data` and is loaded by `src/data_loader_iris.py`.

## Models and Training

`src/iris_dataset.py` contains training and tuning logic for the following classifiers:
- K-Nearest Neighbors (`KNeighborsClassifier`)
- Support Vector Machine (`SVC`)
- Logistic Regression (`LogisticRegression`)
- Random Forest (`RandomForestClassifier`)
- Multi-layer Perceptron (`MLPClassifier`)
- XGBoost (`XGBClassifier`)

The notebook is the primary work in this project; `src/iris_dataset.py` holds a crude form of the notebook code for reference and reuse.

All models are trained on the two features used in the notebook and script:
- `Petal Length`
- `Petal Width`

## Evaluation Results

- All trained models significantly outperform the baseline DummyClassifier (~33% accuracy)
- Several models achieve near-perfect classification performance
- Cross-validation and test scores are closely aligned, indicating minimal overfitting
- Misclassifications occur only between Versicolor and Virginica
- Setosa is perfectly separable from the other classes

Despite similar performance, simpler models (e.g., Logistic Regression, KNN) are preferred due to interpretability and lower complexity.

## Limitations

- The dataset is small and well-separated, so high accuracy is expected for many classifiers.
- Differences between models are minimal
- Results may not generalize to more complex, real-world datasets
- The project focuses on model tuning and evaluation rather than on a production-ready pipeline.
 




