import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load(r'A:\4 project\random_forest_model.pkl')
scaler = joblib.load(r'A:\4 project\scaler.pkl')

# Expected column order (based on your trained model)
expected_columns = [
    'Age', 'MonthlyIncome', 'JobSatisfaction', 'YearsAtCompany',
    'Department_Human Resources', 'Department_Research & Development', 'Department_Sales',
    'MaritalStatus_Divorced', 'MaritalStatus_Married', 'MaritalStatus_Single',
    'OverTime_No', 'OverTime_Yes'
]

numerical_cols = ['Age', 'MonthlyIncome', 'JobSatisfaction', 'YearsAtCompany']

# Preprocessing function
def preprocess_input(input_df):
    # One-hot encode categorical columns
    input_df = pd.get_dummies(input_df, columns=['Department', 'MaritalStatus', 'OverTime'], drop_first=False)

    # Add missing columns and fill with 0
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training
    input_df = input_df[expected_columns]

    # Scale numerical columns
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    return input_df

# Streamlit UI
st.title("💼 Employee Attrition Prediction")
st.markdown("Predict whether an employee is likely to leave the organization.")

# Collect input
age = st.number_input("Age", min_value=18, max_value=65, value=30)
department = st.selectbox("Department", ['Human Resources', 'Research & Development', 'Sales'])
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=4000)
job_satisfaction = st.slider("Job Satisfaction (1 = Low, 4 = High)", 1, 4, 3)
years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
marital_status = st.selectbox("Marital Status", ['Divorced', 'Married', 'Single'])
overtime = st.selectbox("OverTime", ['Yes', 'No'])

# Create DataFrame from input
input_data = pd.DataFrame({
    'Age': [age],
    'Department': [department],
    'MonthlyIncome': [monthly_income],
    'JobSatisfaction': [job_satisfaction],
    'YearsAtCompany': [years_at_company],
    'MaritalStatus': [marital_status],
    'OverTime': [overtime]
})

# Prediction
if st.button("Predict Attrition"):
    try:
        processed_input = preprocess_input(input_data)
        prediction = model.predict(processed_input)

        if prediction[0] == 1:
            st.error("⚠️ The employee is likely to leave (Attrition = Yes).")
        else:
            st.success("✅ The employee is likely to stay (Attrition = No).")
    except Exception as e:
        st.error(f"An error occurred: {e}")
