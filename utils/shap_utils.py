import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def get_shap_explainer(model):
    return shap.TreeExplainer(model)

def get_shap_values(explainer, X):
    shap_values = explainer.shap_values(X)
    return shap_values



def generate_force_plot_image(shap_vals, X, threshold=0.05):
    shap_values_instance = shap_vals[0, :, 1]   # class 1, first row
    feature_names = X.columns.tolist()

    # Filter features by threshold
    filtered = [
        (feature_names[i], shap_values_instance[i])
        for i in range(len(shap_values_instance))
        if abs(shap_values_instance[i]) >= threshold
    ]

    if not filtered:
        return None  # Or return a placeholder image

    # Sort by impact
    filtered.sort(key=lambda x: abs(x[1]), reverse=True)

    names = [f[0] for f in filtered]
    values = [f[1] for f in filtered]
    colors = ['#e74c3c' if val > 0 else '#3498db' for val in values]  # Red = toward fraud, Blue = away

    # Plot
    plt.figure(figsize=(8, len(filtered) * 0.4 + 1))
    bars = plt.barh(names, values, color=colors)
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel("Feature Contribution to Fraud Prediction")
    plt.title("SHAP Feature Impact")
    # Add color legend note
    plt.figtext(0.5, -0.15, "🔴 Red = contributes to suspicious\n🔵 Blue = contributes to normal", 
                ha="center", fontsize=10, color="gray")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return img_base64

def label_confidence(prob):
    if prob < 0.75:
        return "Low"
    elif prob < 0.9:
        return "Medium"
    else:
        return "High"
