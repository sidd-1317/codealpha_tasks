# CodeAlpha Data Science Internship - Task 1
# Iris Flower Classification

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
df.drop(columns=["target"], inplace=True)

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())

# 2. Exploratory Data Analysis
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="sepal length (cm)",
    y="sepal width (cm)",
    hue="species"
)
plt.title("Iris Species by Sepal Measurements")
plt.tight_layout()
plt.savefig("iris_scatter.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df.drop(columns=["species"]))
plt.title("Distribution of Iris Measurements")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("iris_boxplot.png", dpi=150)
plt.show()

# 3. Prepare features and target
X = df.drop(columns=["species"])
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Train classification model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=iris.target_names)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Iris Classification Confusion Matrix")
plt.tight_layout()
plt.savefig("iris_confusion_matrix.png", dpi=150)
plt.show()

# 6. Example prediction
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)[0]
print("\nExample prediction:", prediction)
