import numpy as np

feature_descriptions = {
    "Vehicle Number Encoded": "the vehicle which might have a suspicious/insufficient history associated",
    "Units": "an unusually high or low fuel volume",
    "Driver ID Encoded": "the driver who might have a suspicious/insufficient history associated",
    "Fuel Type_PETROL": "an inconsistent type of fuel used",
    "Amount": "a high amount spent in a single transaction",
    "Actual Odometer": "a suspicious odometer reading",
    "Gallons Over": "an unexpectedly high fuel consumption compared to past transactions",
    "Transaction_Year": "a rare or unexpected transaction year",
    "Transaction_Weekday": "an unusual transaction happening on a non-typical day",
    "Is_Weekend": "a transaction occurring at an unusual time on a weekend",
    "Transaction_Hour_sin": "a transaction happening at an unusual hour",
    "Transaction_Hour_cos": "a transaction occurring at a time that is statistically rare",
    "Transaction_Day_sin": "an unexpected transaction day pattern",
    "Transaction_Day_cos": "a transaction occurring at an uncommon time of the month",
    "Transaction_Month_sin": "a transaction happening in an unexpected season",
    "Transaction_Month_cos": "a transaction occurring at an uncommon time in the year"
}


def generate_explanation(shap_values, data, threshold=0.078):
    shap_values = shap_values[0, :, 1]  # class 1 SHAP values
    feature_names = list(data.columns)

    contributing = []
    counter_evidence = []

    for i in range(len(shap_values)):
        value = shap_values[i]
        name = feature_names[i]
        if value >= threshold:
            contributing.append((name, value))
        elif value <= -0.05:
            counter_evidence.append((name, value))

    # Sort by absolute value of SHAP
    contributing.sort(key=lambda x: abs(x[1]), reverse=True)
    counter_evidence.sort(key=lambda x: abs(x[1]), reverse=True)

    # Main explanation
    if not contributing:
        explanation = "This transaction was flagged, but no single feature stood out as significantly contributing."
    else:
        reasons = [f"due to {feature_descriptions.get(f[0], f[0])}" for f in contributing]
        explanation = "This transaction was flagged as suspicious " + ", ".join(reasons) + "."

    # Add caution if there's strong negative evidence
    if counter_evidence:
        counter_feature = counter_evidence[0][0]  # Just take the top one
        explanation += f" However, {counter_feature} suggests the transaction could be falsely flagged and prompts further investigation."

    return explanation
