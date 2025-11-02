import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import streamlit as st


st.set_page_config(page_title="SSC Pass Rate Predictor", page_icon="📊", layout="centered")
st.title("📘 SSC Pass Rate Predictor (Bangladesh)")
st.write("Predict future SSC pass rates using Polynomial Regression!")


data = {
    'Year': [2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025],
    'Pass_Rate': [35.22, 42.18, 36.85, 58.27, 62.22, 57.37, 72.18, 67.41, 78.19, 82.31, 86.32, 89, 91.34, 87.84, 88.29, 82.35, 77.77, 82.20, 82.87, 83.14, 87.44, 86.39, 83.45, 68.45]
}
df = pd.DataFrame(data)

X = df[['Year']]
y = df['Pass_Rate']

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

future_years = np.array([2026, 2027, 2028, 2029, 2030]).reshape(-1, 1)
future_poly = poly.transform(future_years)
future_predictions = model.predict(future_poly)


st.subheader("📈 Future Pass Rate Predictions (2026–2030)")
prediction_df = pd.DataFrame({
    "Year": future_years.flatten(),
    "Predicted Pass Rate (%)": [round(p, 2) for p in future_predictions]
})
st.table(prediction_df)


fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(X, y, color='#1f77b4', label='Actual Data', s=80)
ax.plot(X, model.predict(X_poly), color='#ff7f0e', linewidth=2, label='Polynomial Fit')
ax.scatter(future_years, future_predictions, color='#2ca02c', s=100, marker='X', label='Predictions')


ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Pass Rate (%)", fontsize=12)
ax.set_title("📊 SSC Pass Rate Prediction (2002–2030)", fontsize=14, weight='bold')
ax.legend(frameon=False)
ax.grid(alpha=0.3, linestyle='--')

st.pyplot(fig)


st.sidebar.markdown("---")  
st.sidebar.header("About the Creator")
st.sidebar.write("👤 **Name:** Fatin Ilham")
st.sidebar.write("📸 [Instagram](https://www.instagram.com/spiritofhonestyyy/)")
st.sidebar.write("📘 [Facebook](https://www.facebook.com/profile.php?id=61572732399921)")
st.sidebar.write("💻 [GitHub](https://github.com/fatin-ilham)")
st.sidebar.write("📧 fatin.ilham@g.bracu.ac.bd")

