import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load and preprocess the dataset
data_path = '/obesity_data.csv'
df = pd.read_csv(data_path)

# Filter out unrealistic BMI values
df = df.loc[(df['BMI'] >= 10) & (df['BMI'] <= 40)]

# Encode categorical variables
df = pd.get_dummies(df, columns=['Gender', 'PhysicalActivityLevel'])

# Split data into features and target
X = df.drop('ObesityCategory', axis=1)
y = df['ObesityCategory']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)

# Train the Decision Tree model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Streamlit app
st.title("Obesity Prediction App")
st.write("""
### Input your details below to find out your weight category.
""")

# Sidebar for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.slider('Age', 10, 80, 25)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    height = st.sidebar.slider('Height (cm)', 120.0, 220.0, 170.0)
    weight = st.sidebar.slider('Weight (kg)', 30.0, 150.0, 70.0)
    physical_activity = st.sidebar.selectbox('Physical Activity Level', [1, 2, 3, 4, 5])
    
    # Calculate BMI
    bmi = weight / ((height / 100) ** 2)

    data = {
        'Age': age,
        'BMI': bmi,
        f'Gender_Female': 1 if gender == 'Female' else 0,
        f'Gender_Male': 1 if gender == 'Male' else 0,
        f'PhysicalActivityLevel_1': 1 if physical_activity == 1 else 0,
        f'PhysicalActivityLevel_2': 1 if physical_activity == 2 else 0,
        f'PhysicalActivityLevel_3': 1 if physical_activity == 3 else 0,
        f'PhysicalActivityLevel_4': 1 if physical_activity == 4 else 0,
        f'PhysicalActivityLevel_5': 1 if physical_activity == 5 else 0,
    }
    return pd.DataFrame(data, index=[0])

# Get user input
df_user = user_input_features()

# Display user input
st.subheader('User Input Parameters')
st.write(df_user)

# Match columns between user input and training data
missing_cols = set(X.columns) - set(df_user.columns)
for col in missing_cols:
    df_user[col] = 0

# Predict obesity category
prediction = model.predict(df_user)
prediction_proba = model.predict_proba(df_user)

# Display prediction
st.subheader('Prediction')
st.write(f"You are classified as: **{prediction[0]}**")

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.write("Note: This prediction is based on a machine learning model and may not be 100% accurate.")
