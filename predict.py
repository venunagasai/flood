import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load trained models
lr_model = joblib.load("logistic_regression_model.pk)
reg_model = joblib.load(linear_regression_model.pkl)

st.title("🌊 Flood & Landslide Prediction App")

# Input form
st.header("📥 Enter Environmental Parameters")

features = [
    "Precipitation", "Temperature", "Max Temperature",
    "Wind Speed", "Cloud Cover", "Humidity"
]

user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"{feature}", min_value=0.0, step=0.1)

if st.button("Predict"):
    input_array = np.array([list(user_input.values())])

    flood_pred = lr_model.predict(input_array)[0]
    landslide_pred = reg_model.predict(input_array)[0]

    flood_text = "🌧️ Flood Risk Detected!" if flood_pred == 1 else "✅ No Flood Risk"
    st.subheader("Prediction Results")
    st.markdown(f"**Flood Prediction:** {flood_text}")
    st.markdown(f"**Estimated Landslides:** {landslide_pred:.2f}")

# Display metrics
st.header("📊 Model Performance")

if st.checkbox("Show Accuracy and Classification Report"):
    # You need original test data for real-time evaluation
    try:
        df = pd.read_csv("flood.csv")
        df['FloodProbability'] = df['FloodProbability'].apply(lambda x: 1 if x >= 0.5 else 0)
        X = df.drop(columns=["FloodProbability", "Landslides"])
        Y = df["FloodProbability"]
        y_pred = lr_model.predict(X)

        st.text("Accuracy: {:.2f}".format(accuracy_score(Y, y_pred)))
        st.text("Classification Report:")
        st.text(classification_report(Y, y_pred))

        st.text("Confusion Matrix:")
        cm = confusion_matrix(Y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error loading or evaluating model: {e}")
