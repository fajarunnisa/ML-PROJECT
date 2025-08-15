import streamlit as st
import joblib

# Load model and scaler
rf = joblib.load("Dry_Eye_Dataset.pkl")
sc = joblib.load("scaler.pkl")

# Load label encoders
le0 = joblib.load('le0.pkl')    # Gender
le = joblib.load('le.pkl')      # Sleep disorder
le1 = joblib.load('le1.pkl')    # Wake up during night
le2 = joblib.load('le2.pkl')    # Feel sleepy during day
le3 = joblib.load('le3.pkl')    # Caffeine consumption
le4 = joblib.load('le4.pkl')    # Alcohol consumption
le5 = joblib.load('le5.pkl')    # Smoking
le6 = joblib.load('le6.pkl')    # Medical issue
le7 = joblib.load('le7.pkl')    # Ongoing medication
le8 = joblib.load('le8.pkl')    # Smart device before bed
le9 = joblib.load('le9.pkl')    # Blue-light filter
le10 = joblib.load('le10.pkl')  # Discomfort Eye-strain
le11 = joblib.load('le11.pkl')  # Redness in eye
le12 = joblib.load('le12.pkl')  # Itchiness/Irritation in eye

st.title("Dry Eye Disease Detection")
st.subheader("Input your health & lifestyle details")

# Numerical Inputs
Age = st.number_input("Age", min_value=0)
Sleep_duration = st.number_input("Sleep duration (hours)", min_value=0.0)
Sleep_quality = st.slider("Sleep quality (1 = Poor, 10 = Excellent)", 1, 10)
Stress_level = st.slider("Stress level (1 = Low, 10 = High)", 1, 10)
Heart_rate = st.number_input("Heart rate (bpm)", min_value=0)
Daily_steps = st.number_input("Daily steps", min_value=0)
Physical_activity = st.number_input("Physical activity (minutes)", min_value=0)
Height = st.number_input("Height (cm)", min_value=0)
Weight = st.number_input("Weight (kg)", min_value=0)
Avg_screen_time = st.number_input("Average screen time (hrs/day)", min_value=0.0)

# Categorical Inputs (select boxes)
Gender = st.selectbox("Gender", le0.classes_)
Sleep_disorder = st.selectbox("Sleep disorder", le.classes_)
Wake_up_night = st.selectbox("Wake up during night", le1.classes_)
Feel_sleepy_day = st.selectbox("Feel sleepy during day", le2.classes_)
Caffeine = st.selectbox("Caffeine consumption", le3.classes_)
Alcohol = st.selectbox("Alcohol consumption", le4.classes_)
Smoking = st.selectbox("Smoking", le5.classes_)
Medical_issue = st.selectbox("Medical issue", le6.classes_)
Medication = st.selectbox("Ongoing medication", le7.classes_)
Smart_device = st.selectbox("Smart device before bed", le8.classes_)
Blue_light = st.selectbox("Blue-light filter", le9.classes_)
Discomfort = st.selectbox("Discomfort/Eye-strain", le10.classes_)
Redness = st.selectbox("Redness in eye", le11.classes_)
Itchiness = st.selectbox("Itchiness/Irritation", le12.classes_)

# Encode categorical features
Gender_enc = le0.transform([Gender])[0]
Sleep_disorder_enc = le.transform([Sleep_disorder])[0]
Wake_up_enc = le1.transform([Wake_up_night])[0]
Sleepy_day_enc = le2.transform([Feel_sleepy_day])[0]
Caffeine_enc = le3.transform([Caffeine])[0]
Alcohol_enc = le4.transform([Alcohol])[0]
Smoking_enc = le5.transform([Smoking])[0]
Medical_enc = le6.transform([Medical_issue])[0]
Medication_enc = le7.transform([Medication])[0]
Smart_device_enc = le8.transform([Smart_device])[0]
Blue_light_enc = le9.transform([Blue_light])[0]
Discomfort_enc = le10.transform([Discomfort])[0]
Redness_enc = le11.transform([Redness])[0]
Itchiness_enc = le12.transform([Itchiness])[0]

# Final input list (24 features)
features = [[
    Age, Sleep_duration, Sleep_quality, Stress_level, Heart_rate,
    Daily_steps, Physical_activity, Height, Weight,
    Sleep_disorder_enc, Wake_up_enc, Sleepy_day_enc, Caffeine_enc,
    Alcohol_enc, Smoking_enc, Medical_enc, Medication_enc,
    Smart_device_enc, Avg_screen_time, Blue_light_enc,
    Discomfort_enc, Redness_enc, Itchiness_enc, Gender_enc
]]

# Scale the features
features_scaled = sc.transform(features)

# Prediction
if st.button("Predict"):
    prediction = rf.predict(features_scaled)[0]
    if prediction == 1:
        st.success("🟠 Dry Eye Disease Detected")
    else :
        st.warning("🟢 No Dry Eye Disease Detected")
    
