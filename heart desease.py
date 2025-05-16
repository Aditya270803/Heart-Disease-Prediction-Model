# Heart Disease Prediction Project
# Author: Aadiv

# 1. Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 2. Load the UCI Heart Disease Dataset
url = "https://raw.githubusercontent.com/aman1108/UCI-Heart-Disease-Dataset/master/heart.csv"
df = pd.read_csv(url)

# 3. Explore Data
print("Shape of dataset:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("Target value counts:\n", df['target'].value_counts())

# 4. Visualize Target Distribution
sns.countplot(x='target', data=df)
plt.title("Heart Disease Distribution (1 = Disease, 0 = No Disease)")
plt.show()

# 5. Feature Selection
X = df.drop('target', axis=1)
y = df['target']

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Model Evaluation Function
def evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\n--- {model_name} ---")
    print("Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
    print("Precision:", round(precision_score(y_test, y_pred)*100, 2), "%")
    print("Recall:", round(recall_score(y_test, y_pred)*100, 2), "%")
    print("F1 Score:", round(f1_score(y_test, y_pred)*100, 2), "%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# 9. Train and Evaluate Models

# Logistic Regression
lr_model = LogisticRegression()
evaluate_model(lr_model, "Logistic Regression")

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
evaluate_model(rf_model, "Random Forest Classifier")

# Support Vector Machine (SVM)
svm_model = SVC(kernel='linear')
evaluate_model(svm_model, "Support Vector Machine")
