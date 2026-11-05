import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="wide"
)

# ---------------------------------------------------------
# Application title and description
# ---------------------------------------------------------
st.title("Breast Cancer Classification")

st.write(
    """
    This application evaluates trained machine learning models on the
    Breast Cancer Wisconsin Diagnostic dataset.

    Upload a test CSV file containing the 30 input features and the
    `diagnosis` target column. Then select a trained model to view its
    performance metrics, confusion matrix, and classification report.

    Target values:

    - **0 = Benign**
    - **1 = Malignant**
    """
)

# ---------------------------------------------------------
# Load saved models and scaler
# ---------------------------------------------------------
@st.cache_resource
def load_saved_files():
    models = {
        "Logistic Regression": joblib.load("model/logistic_model.pkl"),
        "Decision Tree": joblib.load("model/decision_tree_model.pkl"),
        "KNN": joblib.load("model/knn_model.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes_model.pkl"),
        "Support Vector Machine": joblib.load("model/svm_model.pkl")
    }

    scaler = joblib.load("model/scaler.pkl")

    return models, scaler

try:
    models, scaler = load_saved_files()

except FileNotFoundError as error:
    st.error(
        "A saved model or scaler file is missing. Make sure all five "
        "model files and scaler.pkl are in the same folder as app.py."
    )
    st.exception(error)
    st.stop()

except Exception as error:
    st.error("The saved model files could not be loaded.")
    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# Upload test CSV
# ---------------------------------------------------------
st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.info(
        "Please upload the test_data.csv file generated from the notebook."
    )
    st.stop()


# ---------------------------------------------------------
# Read and validate uploaded data
# ---------------------------------------------------------
try:
    test_data = pd.read_csv(uploaded_file)

except Exception as error:
    st.error("The uploaded CSV file could not be read.")
    st.exception(error)
    st.stop()


st.success("Test dataset uploaded successfully.")

st.subheader("Uploaded Data Preview")
st.dataframe(test_data.head(), use_container_width=True)

st.write("Number of rows:", test_data.shape[0])
st.write("Number of columns:", test_data.shape[1])


# ---------------------------------------------------------
# Check target column
# ---------------------------------------------------------
if "diagnosis" not in test_data.columns:
    st.error(
        "The uploaded CSV must contain the target column named 'diagnosis'."
    )
    st.stop()


# ---------------------------------------------------------
# Separate features and target
# ---------------------------------------------------------
X_uploaded = test_data.drop("diagnosis", axis=1)
y_uploaded = test_data["diagnosis"]


# ---------------------------------------------------------
# Validate target values
# ---------------------------------------------------------
if y_uploaded.dtype == "object":
    y_uploaded = y_uploaded.map({
        "B": 0,
        "M": 1,
        "Benign": 0,
        "Malignant": 1
    })


if y_uploaded.isnull().any():
    st.error(
        "The diagnosis column contains invalid or missing values. "
        "Use 0 and 1, or B and M, for the target values."
    )
    st.stop()


y_uploaded = y_uploaded.astype(int)


if not set(y_uploaded.unique()).issubset({0, 1}):
    st.error(
        "The diagnosis column must contain only 0 and 1."
    )
    st.stop()


# ---------------------------------------------------------
# Check for missing feature values
# ---------------------------------------------------------
if X_uploaded.isnull().any().any():
    st.error(
        "The uploaded dataset contains missing feature values. "
        "Please remove or fill the missing values before evaluation."
    )
    st.stop()


# ---------------------------------------------------------
# Ensure feature columns match training columns
# ---------------------------------------------------------
if hasattr(scaler, "feature_names_in_"):
    expected_features = list(scaler.feature_names_in_)

    missing_features = [
        feature
        for feature in expected_features
        if feature not in X_uploaded.columns
    ]

    extra_features = [
        feature
        for feature in X_uploaded.columns
        if feature not in expected_features
    ]

    if missing_features:
        st.error(
            "The uploaded CSV is missing the following required features:"
        )
        st.write(missing_features)
        st.stop()

    if extra_features:
        st.warning(
            "The following extra columns will not be used:"
        )
        st.write(extra_features)

    X_uploaded = X_uploaded[expected_features]


# ---------------------------------------------------------
# Convert all features to numeric
# ---------------------------------------------------------
try:
    X_uploaded = X_uploaded.apply(
        pd.to_numeric,
        errors="raise"
    )

except Exception:
    st.error(
        "All input feature columns must contain numerical values."
    )
    st.stop()


# ---------------------------------------------------------
# Model selection
# ---------------------------------------------------------
st.header("2. Select Classification Model")

selected_model_name = st.selectbox(
    "Select a trained model",
    list(models.keys())
)

selected_model = models[selected_model_name]


# ---------------------------------------------------------
# Prepare input based on selected model
# ---------------------------------------------------------
scaled_models = [
    "Logistic Regression",
    "KNN",
    "Support Vector Machine"
]

if selected_model_name in scaled_models:
    X_for_prediction = scaler.transform(X_uploaded)

else:
    X_for_prediction = X_uploaded


# ---------------------------------------------------------
# Evaluate selected model
# ---------------------------------------------------------
st.header("3. Model Evaluation")

try:
    y_prediction = selected_model.predict(X_for_prediction)

    if hasattr(selected_model, "predict_proba"):
        y_probability = selected_model.predict_proba(
            X_for_prediction
        )[:, 1]

    elif hasattr(selected_model, "decision_function"):
        y_probability = selected_model.decision_function(
            X_for_prediction
        )

    else:
        y_probability = y_prediction

except Exception as error:
    st.error(
        "An error occurred while making predictions. "
        "Check that the uploaded CSV has the correct feature columns."
    )
    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# Calculate required metrics
# ---------------------------------------------------------
accuracy = accuracy_score(
    y_uploaded,
    y_prediction
)

precision = precision_score(
    y_uploaded,
    y_prediction,
    zero_division=0
)

recall = recall_score(
    y_uploaded,
    y_prediction,
    zero_division=0
)

f1 = f1_score(
    y_uploaded,
    y_prediction,
    zero_division=0
)

mcc = matthews_corrcoef(
    y_uploaded,
    y_prediction
)


if y_uploaded.nunique() == 2:
    auc = roc_auc_score(
        y_uploaded,
        y_probability
    )

    auc_display = f"{auc:.4f}"

else:
    auc = None
    auc_display = "Not available"


# ---------------------------------------------------------
# Display metrics
# ---------------------------------------------------------
st.subheader(f"{selected_model_name} Performance")

first_row = st.columns(3)

first_row[0].metric(
    "Accuracy",
    f"{accuracy:.4f}"
)

first_row[1].metric(
    "AUC",
    auc_display
)

first_row[2].metric(
    "Precision",
    f"{precision:.4f}"
)


second_row = st.columns(3)

second_row[0].metric(
    "Recall",
    f"{recall:.4f}"
)

second_row[1].metric(
    "F1 Score",
    f"{f1:.4f}"
)

second_row[2].metric(
    "MCC",
    f"{mcc:.4f}"
)


# ---------------------------------------------------------
# Metrics table
# ---------------------------------------------------------
st.subheader("Metrics Summary")

metrics_table = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC"
    ],
    "Value": [
        round(accuracy, 4),
        round(auc, 4) if auc is not None else "Not available",
        round(precision, 4),
        round(recall, 4),
        round(f1, 4),
        round(mcc, 4)
    ]
})

st.dataframe(
    metrics_table,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------
st.header("4. Confusion Matrix")

confusion_matrix_result = confusion_matrix(
    y_uploaded,
    y_prediction,
    labels=[0, 1]
)

figure, axis = plt.subplots(figsize=(6, 5))

display = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix_result,
    display_labels=["Benign", "Malignant"]
)

display.plot(
    cmap="Blues",
    ax=axis,
    colorbar=False
)

axis.set_title(
    f"Confusion Matrix: {selected_model_name}"
)

st.pyplot(figure)

plt.close(figure)


# ---------------------------------------------------------
# Classification report
# ---------------------------------------------------------
st.header("5. Classification Report")

report = classification_report(
    y_uploaded,
    y_prediction,
    labels=[0, 1],
    target_names=["Benign", "Malignant"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# ---------------------------------------------------------
# Prediction results
# ---------------------------------------------------------
st.header("6. Prediction Results")

prediction_results = pd.DataFrame({
    "Actual Diagnosis": y_uploaded.values,
    "Predicted Diagnosis": y_prediction
})

prediction_results["Actual Label"] = prediction_results[
    "Actual Diagnosis"
].map({
    0: "Benign",
    1: "Malignant"
})

prediction_results["Predicted Label"] = prediction_results[
    "Predicted Diagnosis"
].map({
    0: "Benign",
    1: "Malignant"
})


st.dataframe(
    prediction_results,
    use_container_width=True
)


# ---------------------------------------------------------
# Optional download of prediction results
# ---------------------------------------------------------
prediction_csv = prediction_results.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Prediction Results",
    data=prediction_csv,
    file_name="prediction_results.csv",
    mime="text/csv"
)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")

st.write(
    "The application uses previously trained and saved models. "
    "No model retraining is performed inside the Streamlit application."
)
