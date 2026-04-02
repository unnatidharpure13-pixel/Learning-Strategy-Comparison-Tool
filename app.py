import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Learning Strategy Comparison Tool",
    layout="wide"
)

# -------------------------
# CUSTOM BACKGROUND STYLE
# -------------------------

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(to right, #eef2f3, #dfe9f3);
    }

    h1 {
        color: #2c3e50;
        text-align: center;
    }

    h2, h3 {
        color: #34495e;
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# TITLE
# -------------------------

st.title("Learning Strategy Comparison Tool")

st.markdown("""
This project compares two AI learning approaches:

**Rote Learning**
- Memorizes training examples

**Inductive Learning**
- Learns patterns and generalizes to new data
""")

st.markdown("---")

# -------------------------
# LOAD DATASET
# -------------------------

data = pd.read_csv("dataset/diabetes.csv")

# -------------------------
# DATASET PREVIEW
# -------------------------

st.subheader("Dataset Preview")

st.dataframe(data.head())
 
st.markdown("---")

# -------------------------
# DATA PREPARATION
# -------------------------

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# ROTE LEARNING MEMORY
# -------------------------

memory = {}

for i in range(len(X_train)):
    memory[tuple(X_train.iloc[i])] = y_train.iloc[i]

def rote_predict(sample):
    return memory.get(tuple(sample), "Unknown")

# -------------------------
# INDUCTIVE MODEL
# -------------------------

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# -------------------------
# FEATURE IMPORTANCE GRAPH
# -------------------------

st.subheader("Factors Influencing Diabetes Prediction")

importance = model.feature_importances_

fig2, ax2 = plt.subplots(figsize=(5,3))

ax2.barh(X.columns, importance)

ax2.set_title("Feature Importance")

col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.pyplot(fig2)

st.markdown("---")

# -------------------------
# INPUT SECTION
# -------------------------

st.header("Diabetes Prediction")

col1, col2 = st.columns(2)

with col1:

    preg = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=0
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        max_value=200,
        value=0
    )

    bp = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=150,
        value=0
    )

    skin = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=0
    )

with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=900,
        value=0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=0.0
    )

    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.0
    )

    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=0
    )

# -------------------------
# PREDICTION
# -------------------------

if st.button("Run Prediction"):

    new_patient = [
        preg,
        glucose,
        bp,
        skin,
        insulin,
        bmi,
        dpf,
        age
    ]

    # Inductive prediction
    inductive_result = model.predict(
        pd.DataFrame([new_patient], columns=X.columns)
    )[0]

    # Rote prediction
    rote_result = rote_predict(new_patient)

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Rote Learning")

        if rote_result == "Unknown":
            st.warning("Unknown (data not memorized)")
        elif rote_result == 1:
            st.error("Diabetes Detected")
        else:
            st.success("No Diabetes")

    with col2:

        st.markdown("### Inductive Learning")

        if inductive_result == 1:
            st.error("Diabetes Detected")
        else:
            st.success("No Diabetes")

    # -------------------------
    # COMPARISON GRAPH
    # -------------------------

    st.subheader("Prediction Comparison")

    fig3, ax3 = plt.subplots(figsize=(4,3))

    methods = ["Rote Learning", "Inductive Learning"]

    rote_val = 1 if rote_result == 1 else 0
    inductive_val = 1 if inductive_result == 1 else 0

    values = [rote_val, inductive_val]

    ax3.bar(methods, values)

    ax3.set_ylabel("Prediction (1 = Diabetes)")

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.pyplot(fig3)

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; font-size:14px; color:gray;">
        Made by Unnati • Learning Strategy Comparison Project
    </div>
    """,
    unsafe_allow_html=True
)