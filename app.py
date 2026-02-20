# ==============================
# ClaimWatch AI - User Input Version
# ==============================

import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
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

# Convert target
if y.dtype == 'object':
    y = y.map({'Y': 1, 'N': 0})

# Encode categorical columns
label_encoders = {}

for column in X.columns:
    if X[column].dtype == 'object':
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))
        label_encoders[column] = le

# Handle Missing Values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply SMOTE only on training data
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train Model
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
# User Input Section
# ==============================

st.header("📝 Enter Claim Details")

user_input = {}

for column in data.drop("fraud_reported", axis=1).columns:
    
    if column in label_encoders:
        # Categorical input
        options = label_encoders[column].classes_
        selected = st.selectbox(f"{column}", options)
        encoded = label_encoders[column].transform([selected])[0]
        user_input[column] = encoded
    else:
        # Numeric input
        value = st.number_input(f"{column}", value=0.0)
        user_input[column] = value


# ==============================
# Prediction
# ==============================

if st.button("🔍 Predict Fraud"):

    input_df = pd.DataFrame([user_input])

    # Apply imputer
    input_df = imputer.transform(input_df)

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(f"⚠ Fraudulent Claim Detected!")
    else:
        st.success("✅ Legitimate Claim")

    st.write(f"Fraud Probability: {probability:.2f}")