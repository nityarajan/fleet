import os
import pandas as pd
import tempfile
from flask import Blueprint, request, render_template, send_file

from utils.preprocess import clean_input_data, preprocess, and_predict

predict_bp = Blueprint("predict", __name__)
TEMP_CLEANED_PATH = "/home/user/test/temp/temp_cleaned.csv"
TEMP_RAW_PATH = "/home/user/test/temp/temp_raw.csv"
@predict_bp.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    try:
        # Accept CSV or Excel
        if file.filename.endswith(".xlsx") or file.filename.endswith(".xls"):
            raw_df = pd.read_excel(file)
        else:
            raw_df = pd.read_csv(file)

        cleaned_df = clean_input_data(raw_df)
        df_preprocessed = preprocess(cleaned_df)
        predictions_df = and_predict(df_preprocessed, cleaned_df)

        # Save for reuse
        cleaned_df.to_csv(TEMP_RAW_PATH, index=False)
        predictions_df.to_csv(TEMP_CLEANED_PATH, index=False)

        # Stats for charts
        total = len(predictions_df)
        flagged = predictions_df["Fraud_Prediction"].sum()

        # Confidence levels
        def label_confidence(prob):
            if prob < 0.75:
                return "Low"
            elif prob < 0.9:
                return "Medium"
            return "High"

        confidence_labels = predictions_df[predictions_df["Fraud_Prediction"] == 1]["Fraud_Prediction_Probability"].apply(label_confidence)
        confidence_counts = confidence_labels.value_counts().to_dict()

        return render_template(
            "dashboard.html",
            total=total,
            flagged=flagged,
            confidence=confidence_counts
        )

    except Exception as e:
        return f"Error: {str(e)}", 500
from flask import send_file

@predict_bp.route("/download-all")
def download_all():
    if not os.path.exists(TEMP_CLEANED_PATH):
        return "No file found", 404
    return send_file(TEMP_CLEANED_PATH, as_attachment=True, download_name="all_predictions.csv")

@predict_bp.route("/download-flagged")
def download_flagged():
    if not os.path.exists(TEMP_CLEANED_PATH):
        return "No file found", 404

    df = pd.read_csv(TEMP_CLEANED_PATH)
    flagged_df = df[df["Fraud_Prediction"] == 1]

    # Save to a temporary file
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    flagged_df.to_csv(temp.name, index=False)

    return send_file(temp.name, as_attachment=True, download_name="flagged_predictions.csv")
