from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "tourism_model.joblib"
st.set_page_config(page_title="Tourism Package Purchase Predictor", page_icon="✈️", layout="centered")
st.title("✈️ Wellness Tourism Package Predictor")
st.write("Estimate whether a customer is likely to purchase the Wellness Tourism Package.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()
if model is None:
    st.error("The trained model is not available yet. Run the GitHub Actions pipeline and ensure tourism_model.joblib is committed.")
    st.stop()

with st.form("customer_form"):
    st.subheader("Customer and interaction details")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        contact = st.selectbox("Type of contact", ["Self Inquiry", "Company Invited"])
        city_tier = st.selectbox("City tier", [1, 2, 3], index=1)
        duration = st.number_input("Duration of pitch (minutes)", min_value=0.0, value=10.0, step=1.0)
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        visitors = st.number_input("Number of persons visiting", min_value=1, max_value=20, value=2)
        followups = st.number_input("Number of follow-ups", min_value=0, max_value=20, value=3)
        product = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    with c2:
        stars = st.selectbox("Preferred property stars", [3, 4, 5], index=1)
        marital = st.selectbox("Marital status", ["Single", "Married", "Divorced"])
        trips = st.number_input("Number of trips per year", min_value=0, max_value=100, value=3)
        passport = st.selectbox("Has passport?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        own_car = st.selectbox("Owns a car?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        children = st.number_input("Number of children visiting", min_value=0, max_value=10, value=0)
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        income = st.number_input("Monthly income", min_value=0.0, value=25000.0, step=1000.0)
        satisfaction = st.selectbox("Pitch satisfaction score", [1, 2, 3, 4, 5], index=2)

    submitted = st.form_submit_button("Predict purchase likelihood", type="primary")

if submitted:
    row = pd.DataFrame([{
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
    }])
    # Use model feature order to avoid column-order mismatches.
    expected = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else row.columns.tolist()
    for col in expected:
        if col not in row.columns:
            row[col] = pd.NA
    row = row[expected]
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1]) if hasattr(model, "predict_proba") else None
    if prediction == 1:
        st.success("Prediction: Likely to purchase the package.")
    else:
        st.info("Prediction: Less likely to purchase the package.")
    if probability is not None:
        st.metric("Estimated purchase probability", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))
    with st.expander("Input data sent to the model"):
        st.dataframe(row, use_container_width=True)
    st.caption("This is a model estimate, not a guarantee of customer behavior.")
