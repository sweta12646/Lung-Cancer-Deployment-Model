
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="LungSense AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# 2. LOAD SAVED MODEL
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "lung_cancer_model.pkl"
COMPARISON_PATH = BASE_DIR / "model_comparison.csv"

if not MODEL_PATH.exists():
    st.error(
        "Model file not found! "
        "Please place lung_cancer_model.pkl "
        "in the same folder as cancer.py."
    )
    st.stop()

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
model_name = bundle["model_name"]
features = bundle["features"]

if COMPARISON_PATH.exists():
    comparison = pd.read_csv(COMPARISON_PATH)
else:
    comparison = pd.DataFrame()


# ==================================================
# 3. CUSTOM DARK MODE CSS
# ==================================================

st.markdown("""
<style>

/* Main Background */

.stApp {
    background-color: #0b1220 !important;
    color: #ffffff !important;
}

/* Main Content */

[data-testid="stMain"] {
    background-color: #0b1220 !important;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #111c2e !important;
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* All Labels */

.stApp label {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* General Text */

.stApp p {
    color: #e2e8f0;
}

/* Headings */

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #ffffff !important;
}

/* Selectbox Background */

.stSelectbox div[data-baseweb="select"] > div {
    background-color: #1e293b !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
}

/* Selectbox Text */

.stSelectbox div[data-baseweb="select"] span {
    color: #ffffff !important;
}

/* Selectbox Arrow */

.stSelectbox div[data-baseweb="select"] svg {
    fill: #ffffff !important;
}

/* Dropdown Options Menu */

div[data-baseweb="popover"] {
    background-color: #1e293b !important;
}

div[role="listbox"] {
    background-color: #1e293b !important;
}

div[role="option"] {
    background-color: #1e293b !important;
    color: #ffffff !important;
}

/* Dropdown Hover */

div[role="option"]:hover {
    background-color: #334155 !important;
    color: #38d9c5 !important;
}

/* Slider */

.stSlider label {
    color: #ffffff !important;
}

/* Slider Value */

.stSlider [data-testid="stThumbValue"] {
    color: #38d9c5 !important;
}

/* Form */

[data-testid="stForm"] {
    background-color: #111c2e !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

/* Form Submit Button */

.stFormSubmitButton button {
    background: linear-gradient(
        90deg,
        #0891b2,
        #20bfae
    ) !important;

    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: bold !important;
    font-size: 17px !important;
    padding: 12px !important;
    width: 100% !important;
}

/* Button Hover */

.stButton button:hover,
.stFormSubmitButton button:hover {
    background-color: #38d9c5 !important;
    color: #071522 !important;
    border: none !important;
}

/* Regular Buttons */

.stButton button {
    background-color: #20bfae !important;
    color: #071522 !important;
    border-radius: 8px !important;
    font-weight: bold !important;
}

/* Main Title */

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #38d9c5 !important;
}

/* Subtitle */

.subtitle {
    color: #cbd5e1 !important;
    font-size: 17px;
}

/* Metric Cards */

.metric-card {
    background: linear-gradient(
        135deg,
        #17283d,
        #123b46
    );

    padding: 22px;
    border-radius: 15px;
    border: 1px solid #28566a;
    text-align: center;
}

/* Metric Values */

.metric-value {
    font-size: 30px;
    font-weight: bold;
    color: #38d9c5 !important;
}

/* Metric Labels */

.metric-label {
    color: #ffffff !important;
    font-size: 14px;
}

/* Section Title */

.section-title {
    color: #38d9c5 !important;
    font-size: 25px;
    font-weight: bold;
}

/* Dataframe */

[data-testid="stDataFrame"] {
    background-color: #111c2e !important;
}

/* Success Message */

[data-testid="stAlert"] {
    color: #ffffff !important;
}

/* Metric Text */

[data-testid="stMetricValue"] {
    color: #38d9c5 !important;
}

[data-testid="stMetricLabel"] {
    color: #ffffff !important;
}

/* Captions */

.stCaption {
    color: #cbd5e1 !important;
}

/* Footer */

.footer {
    color: #94a3b8 !important;
    text-align: center;
    font-size: 14px;
    padding: 15px;
}

/* Divider */

hr {
    border-color: #334155 !important;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 4. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.markdown(
    """
    <h1 style="color:#38d9c5;">
        🫁 LungSense AI
    </h1>
    """,
    unsafe_allow_html=True
)

st.sidebar.caption(
    "Smart Respiratory Health Insights"
)

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Health Assessment",
        "Model Performance",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Educational machine-learning project. "
    "Not a medical diagnostic system."
)


# ==================================================
# 5. OVERVIEW PAGE
# ==================================================

if page == "Overview":

    st.markdown(
        '<div class="main-title">'
        'Welcome to LungSense AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore survey patterns and machine-learning predictions.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">309</div>
            <div class="metric-label">Original Dataset Records</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">15</div>
            <div class="metric-label">Input Features</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">6</div>
            <div class="metric-label">ML Algorithms Evaluated</div>
        </div>
        """, unsafe_allow_html=True)
    st.write("")

    st.markdown(
        '<div class="section-title">'
        'Explore the Application'
        '</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:

        st.info(
            "**Health Assessment**\n\n"
            "Enter survey information and explore "
            "the machine-learning prediction."
        )

    with right:

        st.info(
            "**Model Performance**\n\n"
            "Compare the performance of six "
            "classification algorithms."
        )

    st.warning(
        "This application is for educational purposes only. "
        "Its predictions do not establish whether someone "
        "has lung cancer."
    )


# ==================================================
# 6. HEALTH ASSESSMENT PAGE
# ==================================================

elif page == "Health Assessment":

    st.markdown(
        '<div class="main-title">'
        'Health Assessment'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter survey responses to explore a '
        'machine-learning prediction.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    with st.form("health_form"):

        st.markdown(
            '<div class="section-title">'
            'Personal Information'
            '</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:

            gender = st.selectbox(
                "Gender",
                ["M", "F"]
            )

        with col2:

            age = st.slider(
                "Age",
                min_value=21,
                max_value=87,
                value=40
            )

        st.markdown("---")

        st.markdown(
            '<div class="section-title">'
            'Survey Responses'
            '</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "Select the original dataset response code "
            "for each survey feature."
        )

        survey_features = [

            "SMOKING",

            "YELLOW_FINGERS",

            "ANXIETY",

            "PEER_PRESSURE",

            "CHRONIC DISEASE",

            "FATIGUE",

            "ALLERGY",

            "WHEEZING",

            "ALCOHOL CONSUMING",

            "COUGHING",

            "SHORTNESS OF BREATH",

            "SWALLOWING DIFFICULTY",

            "CHEST PAIN"

        ]

        responses = {}

        col1, col2 = st.columns(2)

        for index, feature in enumerate(survey_features):

            with (col1 if index % 2 == 0 else col2):

                responses[feature] = st.selectbox(

                    feature.replace("_", " ").title(),

                    options=[1, 2],

                    format_func=lambda x: f"Code {x}",

                    key=f"input_{feature}"

                )

        st.markdown("---")

        submitted = st.form_submit_button(
            "🫁 Generate Prediction",
            use_container_width=True
        )

    # ----------------------------------------------
    # PREDICTION
    # ----------------------------------------------

    if submitted:

        user_input = {

            "GENDER": gender,

            "AGE": age,

            **responses

        }

        # Match input column names with saved model.
        # This also handles trailing spaces in old model features.

        normalized_input = {
            str(key).strip(): value
            for key, value in user_input.items()
        }

        missing_features = [
            feature for feature in features
            if str(feature).strip() not in normalized_input
        ]

        if missing_features:

            st.error(
                "Missing input features: "
                + ", ".join(missing_features)
            )

        else:

            input_df = pd.DataFrame([
                {
                    feature: normalized_input[str(feature).strip()]
                    for feature in features
                }
            ])

            try:

                prediction = model.predict(input_df)[0]

                st.markdown("---")

                st.markdown(
                    '<div class="section-title">'
                    'Prediction Result'
                    '</div>',
                    unsafe_allow_html=True
                )

                if prediction == "YES":

                    st.error(
                        "⚠️ The model predicts: YES"
                    )

                else:

                    st.success(
                        "The model predicts: NO"
                    )

                st.caption(
                    "This is a machine-learning classification "
                    "output, not a medical diagnosis."
                )

                # Prediction probability

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(
                        input_df
                    )[0]

                    classes = model.classes_

                    confidence = probabilities[
                        list(classes).index(prediction)
                    ] * 100

                    st.metric(
                        "Model Prediction Confidence",
                        f"{confidence:.2f}%"
                    )

                    st.caption(
                        "This percentage represents the model's "
                        "output confidence, not a person's "
                        "actual probability of having cancer."
                    )

            except Exception as e:

                st.error(
                    f"Prediction Error: {e}"
                )


# ==================================================
# 7. MODEL PERFORMANCE PAGE
# ==================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="main-title">'
        'Model Performance'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Compare the six trained classification models.'
        '</div>',
        unsafe_allow_html=True
    )

    if not comparison.empty:

        st.markdown(
            '<div class="section-title">'
            'Model Comparison Table'
            '</div>',
            unsafe_allow_html=True
        )

        st.dataframe(

            comparison.style.format({

                "Accuracy": "{:.3f}",

                "Precision": "{:.3f}",

                "Recall": "{:.3f}",

                "F1 Score": "{:.3f}"

            }),

            use_container_width=True

        )

        st.write("")

        metric = st.selectbox(

            "Choose Evaluation Metric",

            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ]

        )

        fig, ax = plt.subplots(figsize=(10, 5))

        fig.patch.set_facecolor("#0b1220")

        ax.set_facecolor("#111c2e")

        sorted_data = comparison.sort_values(
            by=metric,
            ascending=False
        )

        ax.bar(

            sorted_data["Model"],

            sorted_data[metric],

            color="#20bfae"

        )

        ax.set_title(
            f"Model Comparison - {metric}",
            color="white",
            fontsize=15
        )

        ax.set_ylabel(
            metric,
            color="white"
        )

        ax.set_xlabel(
            "Machine Learning Models",
            color="white"
        )

        ax.tick_params(
            axis="x",
            rotation=35,
            colors="white"
        )

        ax.tick_params(
            axis="y",
            colors="white"
        )

        ax.set_ylim(0, 1.1)

        ax.spines["bottom"].set_color("#475569")
        ax.spines["left"].set_color("#475569")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(
            axis="y",
            alpha=0.2
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.success(
            f"Saved Best Model: {model_name}"
        )

    else:

        st.warning(
            "Model comparison file was not found."
        )


# ==================================================
# 8. ABOUT PROJECT PAGE
# ==================================================

elif page == "About Project":

    st.markdown(
        '<div class="main-title">'
        'About LungSense AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Machine Learning Based Lung Cancer Classification Project'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown("""
    ### Project Overview

    LungSense AI is an educational machine-learning project
    developed using Python and Streamlit.

    It explores survey-based lung cancer classification data
    and demonstrates how different machine-learning algorithms
    can be trained and compared.

    ---

    ### Machine Learning Algorithms

    - Decision Tree Classifier
    - Random Forest Classifier
    - Logistic Regression
    - K-Nearest Neighbors
    - Support Vector Machine
    - Gradient Boosting Classifier

    ---

    ### Technologies Used

    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Matplotlib
    - Streamlit
    - Joblib

    ---

    ### Project Objective

    To demonstrate data preprocessing, model training,
    classification evaluation, and interactive presentation
    of machine-learning outputs.
    """)

    st.warning(
        "Medical Disclaimer: This project is not clinically "
        "validated. It cannot diagnose, screen for, or rule out "
        "lung cancer. Do not use its predictions to make "
        "healthcare decisions. Consult a qualified healthcare "
        "professional for medical concerns."
    )


# ==================================================
# 9. FOOTER
# ==================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        LungSense AI |
        Machine Learning Project |
        Educational Use Only
    </div>
    """,
    unsafe_allow_html=True
)
