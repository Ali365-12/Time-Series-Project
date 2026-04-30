import streamlit as st
import pandas as pd
import numpy as np

st.title("Hello, Streamlit!")
st.write("This is your first Streamlit app.")

st.header("1. Text")
st.write("Use st.write() to display anything.")

st.header("2. User input")
selected_date = st.date_input("Pick a date")
st.write(f"You selected: {selected_date}")

model_choice = st.selectbox("Choose a model", ["XGBoost", "ARIMA", "Prophet", "Holt-Winters"])
st.write(f"You chose: {model_choice}")

days_ahead = st.slider("Days to forecast", min_value=1, max_value=30, value=7)
st.write(f"Forecasting {days_ahead} days ahead.")

st.header("3. Button")
if st.button("Click me"):
    st.success("Button clicked!")

st.header("4. Chart")
sample = pd.DataFrame({
    "Day": range(1, 31),
    "Sales": np.random.randint(300, 700, 30)
}).set_index("Day")
st.line_chart(sample)
