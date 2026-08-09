# CodeAlpha Task 1 — Iris Flower Classification

## Objective
Build a machine-learning classification model that predicts the species of an Iris flower from four measurements: sepal length, sepal width, petal length and petal width.

## Tech Stack
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Random Forest Classifier

## Dataset
This project uses Scikit-learn's built-in Iris dataset, so no CSV download is required.

Official documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html

The dataset contains 150 samples, 3 classes and 4 numeric features.

## How to Run
```bash
python -m pip install -r requirements.txt
python iris_classification.py
```

The program creates:
- `iris_scatter.png`
- `iris_boxplot.png`
- `iris_confusion_matrix.png`

## Workflow
1. Load the Iris dataset.
2. Inspect the data and missing values.
3. Perform exploratory data analysis.
4. Split the data into training and testing sets.
5. Train a Random Forest classifier.
6. Measure accuracy and generate a classification report.
7. Display a confusion matrix.
8. Test one example flower.

## Expected Result
Because the exact result depends on the software version and split, do not hard-code an accuracy value in the report. Use the accuracy printed by your own run.

## GitHub Repository Name
`CodeAlpha_Iris_Flower_Classification`
