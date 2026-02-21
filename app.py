import streamlit as st
import pandas as pd
import joblib

# Load models
try:
    model = joblib.load('calorie_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
except FileNotFoundError:
    st.error("Model files not found!")
    st.stop()

st.set_page_config(page_title="AI Fitness Tracker", page_icon="💪")
st.title("💪 AI-Powered Fitness Analytics")
st.write("Predict your calorie burn using Machine Learning.")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=15, max_value=80, value=25)
    weight = st.number_input("Weight (kg)", min_value=40.0, max_value=150.0, value=70.0)
    duration = st.slider("Workout Duration (hours)", min_value=0.2, max_value=3.0, value=1.0)
with col2:
    avg_bpm = st.slider("Average Heart Rate (BPM)", min_value=80, max_value=200, value=140)
    workout_type = st.selectbox("Workout Type", ['Cardio', 'Strength', 'Yoga', 'HIIT'])
    gender = st.selectbox("Gender", ['Male', 'Female'])

if st.button("🔥 Predict Calories Burned", type="primary"):
    input_data = {
        'Age': age, 'Weight (kg)': weight, 'Avg_BPM': avg_bpm, 'Session_Duration (hours)': duration,
        'Height (m)': 1.75, 'Max_BPM': 180, 'Resting_BPM': 70, 'Fat_Percentage': 20,
        'Water_Intake (liters)': 2.5, 'Workout_Frequency (days/week)': 4, 'Experience_Level': 2,
        'BMI': 22.8, 'Heart_Rate_Reserve': 110,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Workout_Type_HIIT': 1 if workout_type == 'HIIT' else 0,
        'Workout_Type_Strength': 1 if workout_type == 'Strength' else 0,
        'Workout_Type_Yoga': 1 if workout_type == 'Yoga' else 0,
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_df)[0]
    st.success(f"### 🎯 Predicted Calorie Burn: **{int(prediction)} kcal**")
    st.balloons()
