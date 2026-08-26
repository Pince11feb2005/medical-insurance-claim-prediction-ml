import streamlit as st
import pandas as pd
import pickle


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Medical Insurance Claim Predictor",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    with open("gradient_boosting_claim_pipeline.pkl", "rb") as file:
        model = pickle.load(file)

    return model


model = load_model()


# ============================================================
# HEADER
# ============================================================

st.title("🏥 Medical Insurance Claim Prediction")
st.markdown(
    """
    ### Predict Medical Insurance Claim Amount

    Enter the customer's information below and the trained
    Gradient Boosting model will estimate the expected insurance claim.
    """
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=39.5,
        step=1.0
    )

    sex = st.selectbox(
        "Sex",
        ["female", "male"]
    )

    weight = st.number_input(
        "Weight",
        min_value=0.0,
        max_value=300.0,
        value=41.0,
        step=1.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=100.0,
        value=27.6,
        step=0.1
    )


with col2:

    hereditary_diseases = st.selectbox(
        "Hereditary Diseases",
        [
            "NoDisease",
            "Alzheimer",
            "Arthritis",
            "Cancer",
            "Diabetes",
            "HeartDisease"
        ]
    )

    no_of_dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )

    smoker = st.selectbox(
        "Smoker",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    diabetes = st.selectbox(
        "Diabetes",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )


with col3:

    city = st.selectbox(
        "City",
        [
            "Minot",
            "New York",
            "Los Angeles",
            "Chicago",
            "Houston",
            "Phoenix",
            "Dallas",
            "San Antonio"
        ]
    )

    bloodpressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=250.0,
        value=58.0,
        step=1.0
    )

    regular_ex = st.selectbox(
        "Regular Exercise",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    job_title = st.text_input(
        "Job Title",
        value="Dancer"
    )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button(
    "🔮 Predict Claim Amount",
    use_container_width=True
):

    new_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "weight": weight,
        "bmi": bmi,
        "hereditary_diseases": hereditary_diseases,
        "no_of_dependents": no_of_dependents,
        "smoker": smoker,
        "city": city,
        "bloodpressure": bloodpressure,
        "diabetes": diabetes,
        "regular_ex": regular_ex,
        "job_title": job_title
    }])

    try:

        prediction = model.predict(new_data)[0]

        st.success("Prediction generated successfully!")

        st.metric(
            label="Estimated Insurance Claim",
            value=f"{prediction:,.2f}"
        )

        st.subheader("📋 Customer Details")

        display_data = new_data.copy()

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {str(e)}"
        )

        st.info(
            "Make sure the categorical values match the categories "
            "used while training the model."
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📌 Model Information")

    st.write(
        """
        **Model:** Gradient Boosting Regressor

        **Preprocessing:**
        - Missing-value imputation
        - One-Hot Encoding
        - Standard Scaling

        **Feature Selection:**
        - SelectKBest
        - 30 selected features

        **Task:**
        Insurance Claim Amount Prediction
        """
    )

    st.divider()

    st.caption(
        "Medical Insurance Claim Prediction"
    )