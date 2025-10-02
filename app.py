from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load models
heart_model = joblib.load('models/heart_desease.pkl')
diabetes_model = joblib.load('models/diabetes_prediition.pkl')
breast_model = joblib.load('models/breast_cancer_model.pkl')

# Load encoders
gender_encoder = joblib.load('models/label_encoder_gender.pkl')
smoking_encoder = joblib.load('models/label_encoder_somking.pkl')
diagnosis_encoder = joblib.load('models/label_encoder.pkl')

# Health tips
tips = {
    "diabetes": "Maintain a balanced diet, exercise regularly, and monitor your blood sugar.",
    "heart": "Exercise, avoid smoking, control cholesterol, blood pressure, and stress.",
    "breast": "Regular checkups, maintain a healthy weight, and avoid excessive alcohol."
}

# Suggested tests
tests = {
    "diabetes": ["Fasting Blood Sugar Test", "HbA1c Test", "Oral Glucose Tolerance Test"],
    "heart": ["ECG", "Echocardiogram", "Stress Test", "Cholesterol Test"],
    "breast": ["Mammogram", "Breast Ultrasound", "Biopsy if needed"]
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    disease = request.form['disease']

    # Collect features depending on disease
    if disease == "heart":
        try:
            age = get_float_value(request.form, 'age')
            sex = 1 if request.form.get('sex') == 'Male' else 0
            cp = int(request.form.get('cp', 0))
            trestbps = get_float_value(request.form, 'trestbps')
            chol = get_float_value(request.form, 'chol')
            fbs = int(request.form.get('fbs', 0))
            restecg = int(request.form.get('restecg', 0))
            thalach = get_float_value(request.form, 'thalach')
            exang = int(request.form.get('exang', 0))
            oldpeak = get_float_value(request.form, 'oldpeak')
            slope = int(request.form.get('slope', 0))
            ca = int(request.form.get('ca', 0))
            thal = int(request.form.get('thal', 0))
            features = [age, sex, cp, trestbps, chol, fbs,
                    restecg, thalach, exang, oldpeak,
                    slope, ca, thal]
            
            
        except Exception as e:
            return f"Error: {str(e)}"
        model = heart_model

    elif disease == "diabetes":
        # Use encoders here
        gender_val = gender_encoder.transform([request.form['gender']])[0]
        smoking_val = smoking_encoder.transform([request.form['smoking_history']])[0]
        features = [
            gender_val,
            float(request.form['age']),
            int(request.form['hypertension']),
            int(request.form['heart_disease']),
            smoking_val,
            float(request.form['bmi']),
            float(request.form['hba1c']),
            float(request.form['glucose'])
        ]
        model = diabetes_model

    else:  # Breast cancer
        diagnosis_val = diagnosis_encoder.transform([request.form['diagnosis']])[0]
        features = [
            diagnosis_val,
            float(request.form['radius_mean']),
            float(request.form['texture_mean']),
            float(request.form['perimeter_mean']),
            float(request.form['area_mean']),
            float(request.form['smoothness_mean']),
            float(request.form['compactness_mean']),
            float(request.form['concavity_mean']),
            float(request.form['concave_points_mean']),
            float(request.form['symmetry_mean']),
            float(request.form['fractal_dimension_mean']),
            float(request.form['radius_se']),
            float(request.form['texture_se']),
            float(request.form['perimeter_se']),
            float(request.form['area_se']),
            float(request.form['smoothness_se']),
            float(request.form['compactness_se']),
            float(request.form['concavity_se']),
            float(request.form['concave_points_se']),
            float(request.form['symmetry_se']),
            float(request.form['fractal_dimension_se']),
            float(request.form['radius_worst']),
            float(request.form['texture_worst']),
            float(request.form['perimeter_worst']),
            float(request.form['area_worst']),
            float(request.form['smoothness_worst']),
            float(request.form['compactness_worst']),
            float(request.form['concavity_worst']),
            float(request.form['concave_points_worst']),
            float(request.form['symmetry_worst']),
            float(request.form['fractal_dimension_worst'])
        ]
        model = breast_model

    prediction = model.predict([features])[0]

    if prediction == 1:
        result = f"High risk of {disease}. {tips[disease]}"
    else:
        result = f"Low risk of {disease}. {tips[disease]}"

    return render_template('index.html', prediction_text=result)

@app.route('/test')
def disease_tests():
    return render_template('test.html', tests=tests)

def get_float_value(form, key, default=0.0):
    value = form.get(key, "").strip()
    if value == "":
        return default  # or raise an error
    try:
        return float(value)
    except ValueError:
        return default
        
if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)



