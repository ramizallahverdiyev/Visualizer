# pages/Model_Builder.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, root_mean_squared_error


st.set_page_config(page_title="Model Builder", layout="wide")
st.title("🧠 Build & Evaluate a Machine Learning Model")

if "df" not in st.session_state or st.session_state.df.empty:
    st.error("Please upload a dataset from the main page first.")
    st.stop()

df = st.session_state.df.copy()

# Target selection
target = st.selectbox("🎯 Select Target Column", df.columns)

if target:
    X = df.drop(columns=[target])
    y = df[target]

    # Handle categorical variables
    X = pd.get_dummies(X, drop_first=True)
    if y.dtype == "object":
        y = pd.factorize(y)[0]  # Encode target if categorical
        is_classification = True
    else:
        is_classification = False

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    st.subheader("🚧 Training Model...")

    if is_classification:
        model = RandomForestClassifier()
    else:
        model = RandomForestRegressor()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    st.subheader("📈 Model Performance")

    if is_classification:
        st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
    else:
        st.write(f"**RMSE:** {root_mean_squared_error(y_test, y_pred):.2f}")

    # Show predictions
    st.subheader("🔮 Predictions on Test Set")
    results = X_test.copy()
    results["Prediction"] = y_pred
    st.dataframe(results.head(10), use_container_width=True)

    # Optional download button
    st.download_button("⬇️ Download Predictions as CSV", results.to_csv(index=False), file_name="predictions.csv")