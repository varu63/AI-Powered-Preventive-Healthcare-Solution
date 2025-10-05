import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Load dataset

df = pd.read_csv("data.csv")

# Features and target
X = df.drop(columns=['id', 'diagnosis'])
y = df['diagnosis']

# Encode labels (B=0, M=1)
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build pipeline (scaler + logistic regression)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000, random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
print("Test accuracy:", pipeline.score(X_test, y_test))

# Save model + label encoder
joblib.dump(pipeline, "breast_cancer_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ Model and encoder saved as 'breast_cancer_model.pkl' and 'label_encoder.pkl'")