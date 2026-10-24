
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "tourism_model.joblib"

st.set_page_config(
    page_title="Tourism Package Purchase Predictor",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Wellness Tourism Package Predictor")
st.write(
    "Estimate whether a customer is likely to purchase "
    "the Wellness Tourism Package."
)

# --------------------------------------------------
# 2. Load trained model
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.stop()

if model is None:
    st.error(
        "The trained model is not available. "
        "Ensure tourism_model.joblib exists in the deployment folder."
    )
    st.stop()

# --------------------------------------------------
# 3. Customer input form
# --------------------------------------------------

with st.form("customer_form"):
    st.subheader("Customer and interaction details")

    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input(
            "Age", min_value=18, max_value=100, value=35
        )
        contact = st.selectbox(
            "Type of contact",
            ["Self Inquiry", "Company Invited"]
        )
        city_tier = st.selectbox("City tier", [1, 2, 3], index=1)
        duration = st.number_input(
            "Duration of pitch (minutes)",
            min_value=0.0, value=10.0, step=1.0
        )
        occupation = st.selectbox(
            "Occupation",
            ["Salaried", "Small Business", "Large Business", "Free Lancer"]
        )
        gender = st.selectbox("Gender", ["Male", "Female"])
        visitors = st.number_input(
            "Number of persons visiting",
            min_value=1, max_value=20, value=2
        )
        followups = st.number_input(
            "Number of follow-ups",
            min_value=0, max_value=20, value=3
        )
        product = st.selectbox(
            "Product pitched",
            ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
        )

    with c2:
        stars = st.selectbox(
            "Preferred property stars", [3, 4, 5], index=1
        )
        marital = st.selectbox(
            "Marital status", ["Single", "Married", "Divorced"]
        )
        trips = st.number_input(
            "Number of trips per year",
            min_value=0, max_value=100, value=3
        )
        passport = st.selectbox(
            "Has passport?",
            [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )
        own_car = st.selectbox(
            "Owns a car?",
            [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )
        children = st.number_input(
            "Number of children visiting",
            min_value=0, max_value=10, value=0
        )
        designation = st.selectbox(
            "Designation",
            ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
        )
        income = st.number_input(
            "Monthly income",
            min_value=0.0, value=25000.0, step=1000.0
        )
        satisfaction = st.selectbox(
            "Pitch satisfaction score", [1, 2, 3, 4, 5], index=2
        )

    submitted = st.form_submit_button(
        "Predict purchase likelihood",
        type="primary"
    )

# --------------------------------------------------
# 4. Prepare input and predict
# --------------------------------------------------

if submitted:
    input_data = {
        "Age": age,
        "TypeofContact": contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": visitors,
        "NumberOfFollowups": followups,
        "ProductPitched": product,
        "PreferredPropertyStar": stars,
        "MaritalStatus": marital,
        "NumberOfTrips": trips,
        "Passport": passport,
        "PitchSatisfactionScore": satisfaction,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": children,
        "Designation": designation,
        "MonthlyIncome": income,
    }

    row = pd.DataFrame([input_data])

    # Match the model's original input column order.
    expected_columns = getattr(model, "feature_names_in_", None)

    if expected_columns is not None:
        expected_columns = list(expected_columns)

        missing_columns = [
            col for col in expected_columns if col not in row.columns
        ]
        extra_columns = [
            col for col in row.columns if col not in expected_columns
        ]

        if missing_columns:
            st.error(
                "The app does not collect all features required by "
                "the trained model."
            )
            st.write("Missing model features:", missing_columns)
            st.info(
                "Add the missing fields to the form, or retrain the "
                "model using the intended input features."
            )
            st.stop()

        if extra_columns:
            st.warning(
                "The following form fields are not used by this model: "
                + ", ".join(extra_columns)
            )

        row = row[expected_columns]

    # Show inputs for troubleshooting
    with st.expander("Input data and types", expanded=False):
        st.dataframe(row, use_container_width=True)
        st.write("Column data types:")
        st.write(row.dtypes.astype(str))

    try:
        # Predict
        prediction = model.predict(row)[0]

        # Obtain probability when supported
        probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(row)[0]
            classes = list(model.classes_)

            if 1 in classes:
                probability = float(probabilities[classes.index(1)])

        if prediction == 1:
            st.success("Prediction: Likely to purchase the package.")
        else:
            st.info("Prediction: Less likely to purchase the package.")

        if probability is not None:
            st.metric(
                "Estimated purchase probability",
                f"{probability:.1%}"
            )
            st.progress(min(max(probability, 0.0), 1.0))

        st.caption(
            "This is a model estimate, not a guarantee of customer behavior."
        )

    except Exception as e:
        st.error("Prediction failed. Details are shown below.")
        st.code(f"{type(e).__name__}: {e}")

        st.write("Input columns:", row.columns.tolist())
        st.write("Input data types:")
        st.write(row.dtypes.astype(str))

        st.warning(
            "If a numeric imputer receives text, the issue is likely "
            "in the saved preprocessing pipeline or its training feature "
            "definitions. Check train.py and rebuild the model if needed."
        )
