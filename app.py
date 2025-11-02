import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import streamlit as st
st.set_page_config(page_title="SSC Pass Rate Predictor", page_icon="📊", layout="centered")

st.title("📘 SSC Pass Rate Predictor (Bangladesh)")
st.write("Predict future SSC pass rates using Polynomial Regression!")




df = pd.read_csv('https://github.com/fatin-ilham/ssc-pass-predictor/blob/main/SSC%20Result%20Trends%20in%20Bangladesh%20(20012025).csv')
X = df[['Year']]
y = df['Pass_Rate']

poly = PolynomialFeatures(degree=3)  
X_poly = poly.fit_transform(X)


model = LinearRegression()
model.fit(X_poly, y)
future_years = np.array([2026, 2027, 2028, 2029, 2030]).reshape(-1, 1)
future_poly = poly.transform(future_years)
future_predictions = model.predict(future_poly)

st.subheader("📈 Future Pass Rate Predictions")


for year, pred in zip(future_years.flatten(), future_predictions):
    print(f"Predicted Pass Rate for {year}: {pred:.2f}%")
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X, y, color='blue', label='Actual Data')
ax.plot(X, model.predict(X_poly), color='red', label='Polynomial Fit')
ax.scatter(future_years, future_predictions, color='green', label='Predictions')
ax.set_xlabel("Year")
ax.set_ylabel("Pass Rate (%)")
ax.set_title("SSC Pass Rate Prediction (Polynomial Regression)")
ax.legend()

st.pyplot(fig)

