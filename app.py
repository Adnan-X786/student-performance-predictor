import streamlit as st
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓"
)

st.title("🎓 Student Performance Predictor")

st.write(
    "Predict student marks using Machine Learning"
)

study_hours = st.slider(
    "Study Hours per Day",
    1,
    12,
    5
)

attendance = st.slider(
    "Attendance %",
    0,
    100,
    75
)

assignments = st.slider(
    "Assignment Completion %",
    0,
    100,
    70
)

if st.button("Predict Score"):

    prediction = model.predict(
        np.array(
            [[study_hours, attendance, assignments]]
        )
    )[0]

    st.success(
        f"Predicted Score: {prediction:.2f}"
    )
