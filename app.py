import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.set_page_config(page_title="Supermart Sales Predictor")
st.title("🛒 Supermart Grocery Sales Prediction")

st.write("🔢 Please enter the following features (label-encoded where applicable):")

category = st.number_input("Category", min_value=0)
sub_category = st.number_input("Sub Category", min_value=0)
city = st.number_input("City", min_value=0)
region = st.number_input("Region", min_value=0)
state = st.number_input("State", min_value=0)
month = st.number_input("Month (encoded)", min_value=0)
month_no = st.slider("Month Number (1-12)", 1, 12)
order_day = st.slider("Order Day (1-31)", 1, 31)
order_month = st.slider("Order Month (1-12)", 1, 12)
order_year = st.number_input("Order Year (e.g. 2016, 2017, 2018)", min_value=2016, max_value=2020)
discount = st.number_input("Discount", 0.0, 1.0, step=0.01)
profit = st.number_input("Profit", -1000.0, 10000.0, step=10.0)

# Make prediction
if st.button("Predict Sales"):
    input_data = np.array([[category, sub_category, city, region, state,
                            month_no, month, order_day, order_month,
                            order_year, discount, profit]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    st.success(f"📊 Predicted Sales: ₹{prediction:.2f}")
