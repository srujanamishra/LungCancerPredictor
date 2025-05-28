import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('lung_cancer_model.pkl')

st.title("Lung Cancer Risk Predictor")

# Function to get user inputs
def user_input_features():
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 1, 100, 50)

    smoking = st.selectbox("Smoking (0=No, 1=Yes)", [0, 1])
    yellow_fingers = st.selectbox("Yellow Fingers (0=No, 1=Yes)", [0, 1])
    anxiety = st.selectbox("Anxiety (0=No, 1=Yes)", [0, 1])
    peer_pressure = st.selectbox("Peer Pressure (0=No, 1=Yes)", [0, 1])
    chronic_disease = st.selectbox("Chronic Disease (0=No, 1=Yes)", [0, 1])
    fatigue = st.selectbox("Fatigue (0=No, 1=Yes)", [0, 1])
    allergy = st.selectbox("Allergy (0=No, 1=Yes)", [0, 1])
    wheezing = st.selectbox("Wheezing (0=No, 1=Yes)", [0, 1])
    alcohol_consuming = st.selectbox("Alcohol Consuming (0=No, 1=Yes)", [0, 1])
    coughing = st.selectbox("Coughing (0=No, 1=Yes)", [0, 1])
    shortness_of_breath = st.selectbox("Shortness of Breath (0=No, 1=Yes)", [0, 1])
    swallowing_difficulty = st.selectbox("Swallowing Difficulty (0=No, 1=Yes)", [0, 1])
    chest_pain = st.selectbox("Chest Pain (0=No, 1=Yes)", [0, 1])

    gender_encoded = 1 if gender == "Male" else 0

    features = np.array([[gender_encoded, age, smoking, yellow_fingers, anxiety, peer_pressure,
                         chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
                         coughing, shortness_of_breath, swallowing_difficulty, chest_pain]])
    return features


input_data = user_input_features()

if st.button("Predict"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]  # Probability of being positive (i.e., lung cancer)

    result = "Positive for Lung Cancer" if prediction[0] == 1 else "Negative for Lung Cancer"
    st.success(f"Prediction: {result}")

    # Determine risk level based on probability
    if probability < 0.2:
        risk_level = "Very Low"
    elif probability < 0.4:
        risk_level = "Low"
    elif probability < 0.6:
        risk_level = "Moderate"
    elif probability < 0.8:
        risk_level = "High"
    else:
        risk_level = "Very High"

    # Properly formatted multi-line f-string
    st.info(
        (
            f" **Risk Assessment: {risk_level}**\n\n"
            f"Based on your responses, the model suggests a **{risk_level.lower()} likelihood** of lung cancer.\n\n"
            f"> This prediction is based on statistical data and is **not a medical diagnosis**. "
            f"If you have any health concerns, please consult a medical professional."
        )
    )
