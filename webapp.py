#Cabe Morrison, EDUV4938189
#Neccessary imports
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("heart.csv")

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = LogisticRegression(max_iter=2000)
regressor.fit(X_train, y_train)

#Streamlit functionality
st.title("Heart Disease Prediction")
st.sidebar.header("Patient Details")

#User input
def get_user_input():
    age = st.sidebar.slider("Age", 0, 100)
    sex = st.sidebar.selectbox("Sex (0 = female, 1 = male)", (0, 1))
    cp = st.sidebar.selectbox("Chest pain type (cp) (0 = typical angina, 1 = atypical angina , 2 = non-anginal pain , 3 = asymptomatic)", (0, 1, 2, 3))
    trestbps = st.sidebar.slider("Resting blood pressure in mm HG (trestbps)", 0, 300)
    chol = st.sidebar.slider("Serum cholesterol in mg/dl (chol)", 0, 600)
    fbs = st.sidebar.selectbox("Fasting blood sugar > 120 mg/dl (fbs) (0 = false, 1 = true)", (0, 1))
    restecg = st.sidebar.selectbox("Resting electrocardiographic results (restecg) (0 = normal, 1 = abnormal, 2 = ventricular hypertrophy)", (0, 1, 2))
    thalach = st.sidebar.slider("Maximum heart rate achieved (thalach)", 0, 250)
    exang = st.sidebar.selectbox("Exercise-induced angina (exang) (0 = false, 1 = true)", (0, 1))
    oldpeak = st.sidebar.slider("ST depression induced by exercise relative to rest (oldpeak)", 0.00, 10.0)
    slope = st.sidebar.selectbox("Slope of peak exercise ST segment (slope) (0 = upsloping, 1 = flat, 2 = downsloping)", (0, 1, 2))
    ca = st.sidebar.selectbox("Number of major vessels coloured by fluoroscopy (ca)", (0, 1, 2, 3, 4))
    thal = st.sidebar.selectbox("Status of the heart (thal) (0 = unknown, 1 = normal, 2 = fixed defect, 3 = reversible defect)", (0, 1, 2, 3))
    
    user_data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    features = pd.DataFrame(user_data, index=[0])
    return features

#Call function
user_input = get_user_input()

st.subheader("Patient Details")
st.write(user_input)

#Prediction
predict = regressor.predict_proba(user_input)

#Display prediction
st.subheader("Prediction Probability (0 = Healthy Patient, 1 = Patient has heart disease)")
st.write(predict)
