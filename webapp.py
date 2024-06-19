import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv('/mnt/data/heart.csv')

# Split into features and target
X = data.drop('target', axis=1)
y = data['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Streamlit app
st.title('Heart Disease Prediction')

st.sidebar.header('Patient Details')

# Function to get user input
def get_user_input():
    age = st.sidebar.slider('Age', int(data['age'].min()), int(data['age'].max()), int(data['age'].mean()))
    sex = st.sidebar.selectbox('Sex', (0, 1))
    cp = st.sidebar.selectbox('Chest Pain Type', (0, 1, 2, 3))
    trestbps = st.sidebar.slider('Resting Blood Pressure', int(data['trestbps'].min()), int(data['trestbps'].max()), int(data['trestbps'].mean()))
    chol = st.sidebar.slider('Serum Cholesterol', int(data['chol'].min()), int(data['chol'].max()), int(data['chol'].mean()))
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', (0, 1))
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', (0, 1, 2))
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', int(data['thalach'].min()), int(data['thalach'].max()), int(data['thalach'].mean()))
    exang = st.sidebar.selectbox('Exercise Induced Angina', (0, 1))
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', float(data['oldpeak'].min()), float(data['oldpeak'].max()), float(data['oldpeak'].mean()))
    slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', (0, 1, 2))
    ca = st.sidebar.selectbox('Number of Major Vessels Colored by Fluoroscopy', (0, 1, 2, 3, 4))
    thal = st.sidebar.selectbox('Thalassemia', (0, 1, 2, 3))
    
    user_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(user_data, index=[0])
    return features

# Get user input
user_input = get_user_input()

# Display user input
st.subheader('Patient Details')
st.write(user_input)

# Predict
prediction = model.predict(user_input)
prediction_proba = model.predict_proba(user_input)

# Display prediction
st.subheader('Prediction')
heart_disease = np.array(['No Heart Disease', 'Heart Disease'])
st.write(heart_disease[prediction])

# Display prediction probability
st.subheader('Prediction Probability')
st.write(prediction_proba)
