import pandas as pd

df = pd.read_csv('F1 Races 2020-2024.csv')

df.head()

df.info()

df.shape

missing_values = df.isnull().sum()

print(missing_values)

features = ['grid','rainy','Driver Average Position (This Year till last race)','Constructor Average Position (This Year till last race)','laps','driver_age']

target = 'points'

X = df[features]
y = df[target]

X.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error

y_pred = model.predict(X_test)

print(y_pred)

print(y_test)

mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f} points")

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("F1 Points Predictor 🏎️")
st.markdown("A simple model to predict F1 driver points.")
st.markdown("---")

st.subheader("Model Performance")
st.write(f"The model's Mean Absolute Error (MAE) is: **{mae:.2f} points**.")

st.subheader("Visualizing Predictions")
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax.set_xlabel("Actual Points")
ax.set_ylabel("Predicted Points")
ax.set_title("Predicted vs. Actual Points")
st.pyplot(fig)
st.markdown("---")

st.subheader("Make a Prediction!")
st.write("Enter the race conditions and driver stats to get a predicted score.")

grid = st.slider("Starting Grid Position", min_value=1, max_value=22, value=10)
rainy = st.checkbox("Was it a rainy race?")
laps = st.slider("Laps Completed", min_value=1, max_value=80, value=50)

if st.button("Predict Points"):
    user_input = np.array([[grid, int(rainy), 0, 0, laps, 0]])
    predicted_points = model.predict(user_input)
    st.markdown(f"### Predicted Points: **{predicted_points[0]:.2f}**")