import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import os

#LOAD AND PREPROCESS DATA
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    # Drop rows without price
    df = df.dropna(subset=["price"])
    return df

df = load_data()

#TRAIN MODEL IF NOT SAVED
MODEL_PATH = "model.pkl"

def train_and_save_model():
    X = df.drop(columns=["price","name","description"])  # drop target & text columns
    y = df["price"]

    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include=["int64","float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    # Preprocessing
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])

    # Model
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model , num_cols , cat_cols, f)

if not os.path.exists(MODEL_PATH):
    train_and_save_model()

with open(MODEL_PATH, "rb") as f:
    model, num_cols, cat_cols = pickle.load(f)


# STREAMLIT APP
st.set_page_config(page_title=" Vehicle Price Prediction App", layout="wide")
st.title("🚗 Vehicle Price Prediction App")

tab1 , tab2 = st.tabs(["🔍Price Prediction", "📊 Dataset Overview"])

with tab1:
    st.header("Enter vehicle details to predict price")
    input_data = {}
    
    car_name = st.text_input("Car Name" , placeholder = "e.g., 2024 Jeep Wagoneer Series II")
    input_data["name"] = car_name

    for col in num_cols:
        if col != "year":
            input_data[col] = st.number_input(f"{col}" , 
                                              value = float(df[col].median()) if not df[col].isnull().all() else 0.0)
        else:
            input_data[col] = st.number_input(f"{col}", value = int(df[col].median()) , step = 1)

    for col in cat_cols:
        if col == "model":
            input_data[col] = st.text_input("Enter car model" , value = str(df["model"].dropna().iloc[0]))
        else:
            options = sorted(df[col].dropna().unique())
            input_data[col] = st.selectbox(col, options)

if st.button("Predict price"):
    model_input = {k: v for k, v in input_data.items() if k != "name"}
    user_df = pd.DataFrame([model_input])
    prediction = model.predict(user_df)[0]

    prediction = np.clip(prediction, df["price"].min(),df["price"].max())

    st.success(f"💰💸Estimated Price for **{car_name or 'your car'}**: **${prediction:,.2f}**")


with tab2:
    st.subheader("Dataset Overview")
    st.dataframe(df.head(20))

    st.subheader("Summary Statistics")
    st.write(df.describe(include="all"))