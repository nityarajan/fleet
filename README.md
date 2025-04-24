

# About this project
This is a demo of the Fleet fraud detection PoC.

# Executive Summary: 

## Understanding the Problem 

Up to 22% of fleet fuel costs are lost to fraud—silently draining profits, disrupting operations, and creating compliance risks. Fleet fraud isn’t always obvious. It doesn’t just happen in large, coordinated schemes—it often hides internally in routine transactions, slipping through the cracks of standard oversight. From unauthorized fuel-ups to subtle mileage manipulations, fraud takes many forms: 

* Misuse: A driver swipes the company fuel card—but the fuel ends up in a personal vehicle. 

* Slippage: A fleet card is used for "fuel," but the receipt includes snacks and other personal purchases. 

* Overfilling: A driver fuels the truck, then discreetly fills an extra container for resale or personal use. 

These small-scale frauds add up quickly, but they often go unnoticed because traditional security measures—such as PIN protection and transaction limits—aren’t designed to catch behavioral anomalies. However, integrating AI can change this. 

## Our Objectives 

This PoC explores an advanced fraud detection system using AI-driven analytics to:  

* ✔ Detect unusual transaction patterns dynamically, beyond basic rule-based alerts, 

* ✔ Identify high-risk transactions & flag potential fraud in real-time  

* ✔ Provide explainability— Each flagged transaction is accompanied by a clear reason as to why it was flagged to help managers take action with confidence.  

* ✔ Reduce financial losses while minimizing false positives

## Approach
![image](https://github.com/user-attachments/assets/648d73b7-5c3f-4bda-a17c-900756848e7e)
The overall approach is to first identify patterns in the real world data: what is the norm, what is an outlier? In addition to logical anomalies. So, what amount spent is suspicious? How many units purchased is too much? Why did the odometer value decrease. These are the lines of questioning we followed during EDA to define a "suspicious transaction". We then move onto labelling the dataset based on these flags, and then training a binary random forest classsifier on these labels. This model is then wrapped in an API using Flask and integrated into the simple web app here. 

### EDA
![image](https://github.com/user-attachments/assets/cbd017d8-28f2-4915-8a3e-306ae81f1d3a)

The goal of the EDA was to explore the threshholds of a suspicious transaction and to clean and prepare the dataset for modelling

### Dataset Labelling

We identified 7 key flags for the labelling process. If any one goes off, the transaction is labelled suspicious 

![image](https://github.com/user-attachments/assets/9fe85b36-4ca1-4c58-8fee-fd5c4d05a855)

### Model building 
After flagging, the dataset was prepared for modelling (encoding and scaling). Additionally, there was a mild class imbalance (35% suspicious transactions -> minority class) and random undersampling was used to balance the classes 
![image](https://github.com/user-attachments/assets/e1986650-1af6-4634-8513-684eab73d30f)


## Final Model

The final model is a Binary Random Forest Regression. The model acheived 84% Accurary (with the flagging system) and an AUC-ROC  of 0.91

# Demo

This demo was built on Firebase Studio, with HTML CSS JS as the front-end and Flask used to wrap the model in a simply API.
The model was imported in a folder /model in a pkl format, alongside the scalers and encoders. However this is not available in this repository due to file size. 
* Routes: contains 2 routes: explain and predict.
    * Predict is called after a csv is uploaded. It uses functions in utils/preprocess.py
    * Explain is called when the explain this transaction button is called after submitting a transaction uid. It uses functions from utils/explain.py and utils/shap_utils.py
 * Templates: contains HTML templates for the 4 key pages: index for the homepage, dashboard for the dashboard summany of the predictions of the csv file, explain for the explanation page and error
 * static: contains css file
 * Utils: contains utility functions for explain and predict routes
 * Furthermore, there are 2 folders that are not available in this repository:
     * temp to store the uploaded csv and prediction csv
     * model which stores the model weights, encoders and scaler in pkl format

# Flask Web App Starter

A Flask starter template as per [these docs](https://flask.palletsprojects.com/en/3.0.x/quickstart/#a-minimal-application).

## Getting Started

Previews should run automatically when starting a workspace.
