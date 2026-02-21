# ==============================
# ClaimWatch AI - User Input Version (With Demo Auto-Fill)
# ==============================

import pandas as pd
import numpy as np
import streamlit as st
import joblib
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer,MissingIndicator
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


# ==============================
# Streamlit Title
# ==============================
st.title("🚀 ClaimWatch AI - Fraud Detection System")


# ==============================
# Load Dataset
# ==============================
@st.cache_data
def load_data():
    return pd.read_csv("insurance_claims.csv")

data = load_data()


# ==============================
# Preprocessing
# ==============================

X = data.drop("fraud_reported", axis=1)
y = data["fraud_reported"]

if y.dtype == 'object':
    y = y.map({'Y': 1, 'N': 0})

label_encoders = {}

for column in X.columns:
    if X[column].dtype == 'object':
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))
        label_encoders[column] = le

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)


# ==============================
# DEMO DATA GENERATOR
# ==============================

def generate_demo_data(fraud=False):

    if fraud:
        # Pick real fraud case from dataset
        fraud_rows = data[data["fraud_reported"] == "Y"]
        sample = fraud_rows.sample(1).drop("fraud_reported", axis=1)

    else:
        # Pick real genuine case
        genuine_rows = data[data["fraud_reported"] == "N"]
        sample = genuine_rows.sample(1).drop("fraud_reported", axis=1)

    demo = {}

    for column in sample.columns:

        if column in label_encoders:
            encoded = label_encoders[column].transform(
                [str(sample.iloc[0][column])]
            )[0]
            demo[column] = encoded
        else:
            demo[column] = float(sample.iloc[0][column])

    return demo


# ==============================
# Auto-Fill Buttons
# ==============================

col1, col2 = st.columns(2)

with col1:
    if st.button("🚨 Generate Fraud Claim"):
        st.session_state.demo_data = generate_demo_data(fraud=True)

with col2:
    if st.button("🎯 Generate Genuine Claim"):
        st.session_state.demo_data = generate_demo_data(fraud=False)


# ==============================
# User Input Section
# ==============================

st.header("📝 Enter Claim Details")

user_input = {}

columns = data.drop("fraud_reported", axis=1).columns

for column in columns:

    if column in label_encoders:
        classes = label_encoders[column].classes_

        default_index = 0

        if "demo_data" in st.session_state:
            encoded_val = st.session_state.demo_data.get(column, 0)
            decoded_val = label_encoders[column].inverse_transform([int(encoded_val)])[0]
            default_index = list(classes).index(decoded_val)

        selected = st.selectbox(f"{column}", classes, index=default_index)
        encoded = label_encoders[column].transform([selected])[0]
        user_input[column] = encoded

    else:
        default_value = 0.0

        if "demo_data" in st.session_state:
            default_value = float(st.session_state.demo_data.get(column, 0.0))

        value = st.number_input(f"{column}", value=default_value)
        user_input[column] = value


# ==============================
# Prediction
# ==============================

if st.button("🔍 Predict Fraud"):

    input_df = pd.DataFrame([user_input])
    input_df = imputer.transform(input_df)

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠ Fraudulent Claim Detected!")
    else:
        st.success("✅ Legitimate Claim")

    st.write(f"Fraud Probability: {probability:.2f}")