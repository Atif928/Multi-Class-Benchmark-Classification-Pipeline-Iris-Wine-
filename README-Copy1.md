# Multi-Class Benchmark Classification Pipeline (Iris Dataset)

## Project Objective

This project implements and compares two supervised machine learning classification algorithms using the Iris dataset:

1. **Logistic Regression**
2. **Decision Tree Classifier**

The pipeline demonstrates data loading, exploratory data analysis (EDA), feature scaling using StandardScaler, an 80/20 train-test split with stratification, model training, evaluation with classification metrics, confusion matrix heatmaps, and decision boundary visualization.

## Dataset

The project uses the Iris dataset from [Seaborn's dataset repository](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv). It contains four numerical features:
- `sepal_length`
- `sepal_width`
- `petal_length`
- `petal_width`

The target variable `species` has three classes:
- `setosa`
- `versicolor`
- `virginica`

A fallback to `sklearn.datasets.load_iris(as_frame=True)` is implemented if CSV loading fails.

## Project Structure

- `multiclass_classification_pipeline.ipynb` - Main Jupyter notebook with complete analysis
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation
- `iris.csv` - Local copy of the dataset (if present)

## Installation

1. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

2. Install dependencies:
```bash
python -m pip install -r requirements.txt
```

## Usage

Launch Jupyter Notebook:
```bash
python -m notebook
```

Then open `multiclass_classification_pipeline.ipynb` and run all cells from top to bottom.

## Features Demonstrated

- Data loading with fallback mechanism
- Data inspection and validation
- Exploratory Data Analysis (count plots, pairplots, correlation heatmaps, distributions)
- Train-test split with stratification
- Scikit-learn Pipelines with StandardScaler
- Model training and prediction
- Performance benchmarking (Accuracy, Macro Precision, Recall, F1-Score)
- Classification reports
- Confusion matrix visualizations
- Decision boundary plots (2D visualization)
- Reproducible results with random seeds

## Notes

- Feature scaling is applied via StandardScaler in pipelines. While Decision Trees don't strictly require scaling, it's kept in the pipeline for consistent preprocessing demonstration.
- Decision boundaries are visualized using only `petal_length` and `petal_width` for 2D plotting - these are separate models trained on those two features only.
- All metrics use macro-averaging for fair evaluation across all three classes.
