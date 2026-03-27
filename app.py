import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Page config
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# 🌈 Colorful CSS
st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #84fab0, #8fd3f4);
}

.main {
    background: linear-gradient(135deg, #fdfbfb, #ebedee);
    padding: 20px;
    border-radius: 20px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.2);
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #ff4b2b;
}

.stButton>button {
    background: linear-gradient(45deg, #36d1dc, #5b86e5);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(45deg, #ff758c, #ff7eb3);
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>❤️ Heart Disease Prediction</div>", unsafe_allow_html=True)

st.write("### 🧑‍⚕️ Enter Patient Details")

# Load data
df = pd.read_csv("heart_disease_data.csv")
X = df.drop("target", axis=1)
y = df["target"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Card UI
st.markdown("<div class='card'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 1, 100, 25)
    sex = st.radio("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0,1,2,3])

with col2:
    trestbps = st.slider("Blood Pressure", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 400, 200)
    thalach = st.slider("Max Heart Rate", 60, 220, 150)

sex_value = 1 if sex == "Male" else 0

# Button
if st.button("🔍 Predict Now"):
    input_data = np.array([[age, sex_value, cp, trestbps, chol, 0, 0, thalach, 0, 0, 0, 0, 0]])
    prediction = model.predict(input_data)

    st.markdown("---")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
        st.progress(80)
    else:
        st.success("✅ Low Risk of Heart Disease")
        st.progress(30)

st.markdown("</div>", unsafe_allow_html=True)