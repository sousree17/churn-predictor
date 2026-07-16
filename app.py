import streamlit as st
import pandas as pd
import joblib

# 1. Load the model and features
model = joblib.load('churn_model.pkl')
features = joblib.load('model_features.pkl')

# 2. Build the visual interface
st.title("AI Customer Churn Predictor 📊")
st.write("Enter customer data below to predict if they are likely to cancel their subscription.")

# 3. Create input fields for the user
tenure = st.slider("Months with company (Tenure)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0)
total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)
contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# 4. Create a predict button
if st.button("Predict Churn Risk"):

    # Create a dictionary filled with zeros for all 30 features the model expects
    input_data = {col: 0 for col in features}

    # Update the specific features we took from the user
    input_data['tenure'] = tenure
    input_data['MonthlyCharges'] = monthly_charges
    input_data['TotalCharges'] = total_charges

    # Handle the one-hot encoded contract type logic
    if contract_type == "One year":
        input_data['Contract_One year'] = 1
    elif contract_type == "Two year":
        input_data['Contract_Two year'] = 1

    # Convert the single customer data into a dataframe
    input_df = pd.DataFrame([input_data])

    # Make the prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Display the results
    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Churn! (Probability: {probability:.0%})")
    else:
        st.success(f"✅ Customer is likely to stay. (Churn Probability: {probability:.0%})")