
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# 1. Load dataset
df = pd.read_csv("dataset/creditcard.csv")

print("Dataset loaded:", df.shape)

# 2. Separate input features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# 4. Create classification model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# 5. Train the model
print("\nTraining model...")
model.fit(X_train, y_train)

# 6. Predict test data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 7. Evaluate model
print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
print(
    "PR-AUC (Average Precision):",
    round(average_precision_score(y_test, y_prob), 4)
)

# 8. Save trained model
joblib.dump(model, "fraud_model.joblib")

print("\nModel saved as fraud_model.joblib")
print("TRAINING COMPLETED SUCCESSFULLY!")