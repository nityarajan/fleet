from flask import Blueprint, request, render_template
import pandas as pd
import joblib
from utils.explain import generate_explanation
from utils.shap_utils import get_shap_explainer, get_shap_values, generate_force_plot_image, label_confidence
from utils.preprocess import preprocess
explain_bp = Blueprint("explain", __name__)
TEMP_CLEANED_PATH = "temp/temp_cleaned.csv"
TEMP_RAW_PATH = "temp/temp_raw.csv"

model = joblib.load("model/rf_fraud_model_final.pkl")
explainer = get_shap_explainer(model)

@explain_bp.route("/explain", methods=["POST"])
def explain_transaction():
    uid = request.form.get("uid")
    if not uid:
        return "No UID provided", 400

    try:
        df = pd.read_csv(TEMP_CLEANED_PATH)
        df_raw = pd.read_csv(TEMP_RAW_PATH)
    except Exception as e:
        return f"Could not load previous predictions: {str(e)}", 500

    if "transaction_uid" not in df.columns:
        return "transaction_uid not found", 400

    row = df[df["transaction_uid"].astype(str) == uid]
    row_raw = df_raw[df_raw["transaction_uid"].astype(str) == uid]
    if row.empty:
        return f"No transaction found with UID {uid}", 404

    try:
        X = preprocess(row)
        X = X[['Units', 'Amount', 'Actual Odometer', 'Gallons Over',
                    'Vehicle Number Encoded', 'Driver ID Encoded', 'Fuel Type_PETROL',
                    'Transaction_Year', 'Transaction_Weekday', 'Is_Weekend',
                    'Transaction_Hour_sin', 'Transaction_Hour_cos',
                    'Transaction_Day_sin', 'Transaction_Day_cos',
                    'Transaction_Month_sin', 'Transaction_Month_cos']]

        shap_vals = get_shap_values(explainer, X)
        
        explanation = generate_explanation(shap_vals, X)

        img_base64 = generate_force_plot_image(shap_vals, X)
        #explanation = "test"
        prediction = int(row["Fraud_Prediction"].values[0])
        prob = float(row["Fraud_Prediction_Probability"].values[0])
        confidence_label = label_confidence(prob)
        if(prediction ==1):
            confidence_label = label_confidence(prob)
            return render_template(
                "explain.html",
                uid=uid,
                prediction=prediction,
                prob=round(prob * 100, 2),
                confidence_label=confidence_label,
                explanation=explanation,
                details=row_raw.to_dict(orient="records")[0],
                shap_image=img_base64
            )
        else:
            confidence_label = label_confidence((1-prob))
            return render_template(
                "explain.html",
                uid=uid,
                prediction=prediction,
                prob=round((1-prob) * 100, 2),
                confidence_label=confidence_label,
                explanation="The transaction is not suspicious",
                details=row_raw.to_dict(orient="records")[0],
                shap_image=img_base64
            )

    except Exception as e:
        return f"Error generating explanation: {str(e)}", 500
