
import time
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import BytesIO
import matplotlib.pyplot as plt

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="🌟 Employee Attrition App", layout="wide")

# ----------------------------
# Session States
# ----------------------------
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if 'logout' not in st.session_state:
    st.session_state.logout = False

if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0

if 'last_activity' not in st.session_state:
    st.session_state.last_activity = time.time()

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# ----------------------------
# Session Timeout
# ----------------------------
session_timeout = 30 * 60  # 30 minutes
if time.time() - st.session_state.last_activity > session_timeout:
    st.session_state.authenticated = False
    st.session_state.logout = True
    st.rerun()

# ----------------------------
# Authentication
# ----------------------------
def login_page():
    st.title("🔐 Employee Attrition Prediction Login")
    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")
    if st.button("🔓 Login"):
        if username == "admin" and password == "password":
            st.session_state.authenticated = True
            st.success("✅ Login successful!")
            st.session_state.last_activity = time.time()
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

def logout():
    st.session_state.authenticated = False
    st.session_state.logout = True

# ----------------------------
# Dark Mode Styling
# ----------------------------
def set_dark_mode():
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        html, body, [class*="css"]  {
            background-color: #181818 !important;
            color: #ffffff !important;
        }
        .stButton > button {
            background-color: #333333;
            color: white;
        }
        .stTextInput input, .stSelectbox div, .stSlider > div {
            background-color: #2e2e2e;
            color: white;
        }
        .stDownloadButton > button {
            background-color: #444;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

# ----------------------------
# Prediction Page
# ----------------------------
def prediction_app():
    st.title("📊 Employee Attrition Prediction")

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)

    st.subheader("📝 Input Employee Information")

    with st.expander("🔧 Customize Input Features", expanded=True):
        def user_input_features():
            col1, col2 = st.columns(2)

            with col1:
                Age = st.slider("📅 Age", 18, 60, 30)
                DailyRate = st.number_input("📈 Daily Rate", 100, 1500, 500)
                DistanceFromHome = st.slider("📍 Distance From Home", 1, 30, 5)
                EmployeeNumber = st.number_input("🆔 Employee Number", 1, 10000, 101)
                HourlyRate = st.number_input("⏱️ Hourly Rate", 10, 100, 50)
                MonthlyIncome = st.number_input("💵 Monthly Income", 1000, 20000, 5000)
                MonthlyRate = st.number_input("💲 Monthly Rate", 1000, 30000, 10000)

            with col2:
                TotalWorkingYears = st.slider("🧮 Total Working Years", 0, 40, 10)
                YearsAtCompany = st.slider("🏢 Years at Company", 0, 30, 5)
                Department = st.selectbox("🏬 Department", ["Sales", "Research & Development", "Human Resources"])
                EnvironmentSatisfaction = st.selectbox("🌿 Environment Satisfaction", ["Low", "Medium", "High", "Very High"])
                JobInvolvement = st.selectbox("💼 Job Involvement", ["Low", "Medium", "High", "Very High"])
                MaritalStatus = st.selectbox("💍 Marital Status", ["Single", "Married", "Divorced"])
                OverTime = st.selectbox("⏳ OverTime", ["Yes", "No"])
                WorkLifeBalance = st.selectbox("⚖️ Work Life Balance", ["Bad", "Good", "Better", "Best"])

            data = {
                "Age": Age,
                "DailyRate": DailyRate,
                "DistanceFromHome": DistanceFromHome,
                "EmployeeNumber": EmployeeNumber,
                "HourlyRate": HourlyRate,
                "MonthlyIncome": MonthlyIncome,
                "MonthlyRate": MonthlyRate,
                "TotalWorkingYears": TotalWorkingYears,
                "YearsAtCompany": YearsAtCompany,
                "Department": Department,
                "EnvironmentSatisfaction": EnvironmentSatisfaction,
                "JobInvolvement": JobInvolvement,
                "MaritalStatus": MaritalStatus,
                "OverTime": OverTime,
                "WorkLifeBalance": WorkLifeBalance
            }
            return pd.DataFrame([data])

        input_df = user_input_features()
        input_data_encoded = pd.get_dummies(input_df)
        input_data_encoded = input_data_encoded.reindex(columns=model_columns, fill_value=0)
        input_scaled = scaler.transform(input_data_encoded)
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

    st.subheader("🔍 Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Employee likely to **LEAVE**.")
    else:
        st.success("✅ Employee likely to **STAY**.")

    st.info(f"📈 Staying Probability: `{prediction_proba[0][0]*100:.2f}%`")
    st.info(f"📉 Leaving Probability: `{prediction_proba[0][1]*100:.2f}%`")

    result_df = input_df.copy()
    result_df['Attrition Prediction'] = ['Yes' if prediction[0] == 1 else 'No']
    result_df['Attrition Probability'] = [round(prediction_proba[0][1], 2)]

    csv = result_df.to_csv(index=False).encode()
    st.download_button("📥 Download Report", data=csv, file_name="prediction_report.csv", mime="text/csv")

    st.session_state.total_predictions += 1

# ----------------------------
# Model Comparison Page
# ----------------------------


def model_comparison():
    st.title("📈 Model Comparison & Evaluation")

    st.subheader("📊 Accuracy Scores of ML Models")

    import pandas as pd

    # Updated accuracy scores
    data = {
        'Algorithms': [
            'Logistic Regression',
            'Random Forest',
            'Support Vector Machine',
            'XGBoost',
            'LightGBM',
            'CatBoost',
            'AdaBoost'
        ],
        'Training Accuracy': [
            0.9291,
            0.8397,
            0.9349,
            1.0000,
            1.0000,
            0.9864,
            0.9009
        ],
        'Testing Accuracy': [
            0.8685,
            0.8390,
            0.8594,
            0.8413,
            0.8526,
            0.8481,
            0.8345
        ]
    }

    df_results = pd.DataFrame(data)

    # Apply styling
    styled_df = df_results.style\
        .set_properties(subset=['Training Accuracy'], **{'background-color': '#D6EAF8', 'color': 'black'})\
        .set_properties(subset=['Testing Accuracy'], **{'background-color': '#FADBD8', 'color': 'black'})\
        .set_properties(subset=['Algorithms'], **{'font-weight': 'bold'})

    # Display styled dataframe
    st.dataframe(styled_df, use_container_width=True)



# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("🧭 Navigation")

if st.session_state.authenticated:
    st.sidebar.markdown("### 👋 Welcome, Admin!")
    st.sidebar.markdown(f"### 📊 Total Predictions Made: `{st.session_state.total_predictions}`")

    st.sidebar.subheader("🌓 Dark Mode")
    st.session_state.dark_mode = st.sidebar.checkbox("Enable Dark Mode")
    set_dark_mode()

    st.sidebar.subheader("🛠️ Help")
    st.sidebar.markdown("For more info, visit the [📄 documentation](https://example.com).")

    st.sidebar.subheader("⭐ Rate Your Experience")
    rating = st.sidebar.slider("Rate the app", 1, 5)
    if rating:
        st.sidebar.markdown(f"🙏 Thanks for rating us {rating}/5!")

    page = st.sidebar.radio("🔀 Go to", ["Prediction", "Model Comparison"])
    st.sidebar.button("🚪 Logout", on_click=logout)

    if page == "Prediction":
        prediction_app()
    elif page == "Model Comparison":
        model_comparison()
else:
    login_page()

if st.session_state.logout:
    st.session_state.logout = False
    st.rerun()

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("<center>Made with ❤️ by Somya Khare</center>", unsafe_allow_html=True)
