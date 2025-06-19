# Ini Error
# import streamlit as st
# import numpy as np
# from joblib import load

# # Load model and scaler
# model = load('model.joblib')
# scaler = load('scaler.pkl')

# st.title("🧠 Stroke Risk Predictor")

# st.markdown("Please answer the following symptoms with **Yes** or **No**, and enter your age.")

# # Create a list of 14 symptoms
# symptoms = [
#     "Chest Pain",
#     "Shortness of Breath",
#     "Irregular Heartbeat",
#     "Fatigue & Weakness",
#     "Dizziness",
#     "Swelling (Edema)",
#     "Pain in Neck/Jaw/Shoulder/Back",
#     "Excessive Sweating",
#     "Persistent Cough",
#     "Nausea/Vomiting",
#     "High Blood Pressure",
#     "Chest Discomfort (Activity)",
#     "Cold Hands/Feet",
#     "Snoring/Sleep Apnea",
#     "Anxiety/Feeling of Doom"
# ]

# for symptom in symptoms:
#     choice = st.selectbox(f"{symptom}:", ["No", "Yes"])
#     symptoms.append(1 if choice == "Yes" else 0)

# # Age input
# age = st.number_input("Age", min_value=0, max_value=120, value=30)
# symptoms.append(age)

# # Predict button
# if st.button("Predict Stroke Risk"):
#     input_array = np.array(symptoms).reshape(1, -1)
#     input_scaled = scaler.transform(input_array)
#     prediction = model.predict(input_scaled)[0]

#     if prediction == 1:
#         st.error("⚠️ Prediction: You may be at risk of stroke.")
#     else:
#         st.success("✅ Prediction: You are likely not at risk of stroke.")

# Run -> python -m streamlit run "gpt.py"
import streamlit as st
import numpy as np
from joblib import load

# Load model and scaler
model = load('model.joblib')
scaler = load('scaler.pkl')

st.title("🧠 Stroke Risk Predictor")
st.markdown("Please answer the following symptoms with **Yes** or **No**, and enter your age.")

# Symptom labels (14 used for the model)
symptom_labels = [
    "Chest Pain",
    "Shortness of Breath",
    "Irregular Heartbeat",
    "Fatigue & Weakness",
    "Dizziness",
    "Swelling (Edema)",
    "Pain in Neck/Jaw/Shoulder/Back",
    "Excessive Sweating",
    "Persistent Cough",
    "Nausea/Vomiting",
    "High Blood Pressure",
    "Chest Discomfort (Activity)",
    "Cold Hands/Feet",
    "Snoring/Sleep Apnea",
    "Anxiety/Feeling of Doom"
]

# Collect user input
user_inputs = []
for idx, symptom in enumerate(symptom_labels):
    choice = st.selectbox(f"{symptom}:", ["No", "Yes"], key=f"symptom_{idx}")
    user_inputs.append(1 if choice == "Yes" else 0)

# Age input
age = st.number_input("Age", min_value=0, max_value=120, value=30)
user_inputs.append(age)

# Predict button
if st.button("Predict Stroke Risk"):
    input_array = np.array(user_inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ Prediction: You may be at risk of stroke.")
    else:
        st.success("✅ Prediction: You are likely not at risk of stroke.")