import pandas as pd

df = pd.read_csv("diabetes_prediction_dataset.csv")

# Clean diabetes column: map to integers
df['diabetes'] = df['diabetes'].astype(str).map({'0': 0, '0f': 0, '1': 1})

# Drop rows with invalid mappings (if any)
df = df.dropna(subset=['diabetes'])
df['diabetes'] = df['diabetes'].astype(int)

from sklearn.preprocessing import LabelEncoder

# Encode gender
le_gender = LabelEncoder()
df['gender'] = le_gender.fit_transform(df['gender'])

# Encode smoking_history
le_smoking = LabelEncoder()
df['smoking_history'] = le_smoking.fit_transform(df['smoking_history'].astype(str))

# Features and target
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Predictions and evaluation
from sklearn.metrics import classification_report, accuracy_score
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

import joblib
joblib.dump(le_gender,"label_encoder_gender.pkl")
joblib.dump(le_smoking,"label_encoder_somking.pkl")
joblib.dump(model,"diabetes_prediition.pkl")
