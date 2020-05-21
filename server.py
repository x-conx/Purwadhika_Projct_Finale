import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, url_for

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('prediction.html')

@app.route('/prediction', methods=["POST", "GET"])
def creditcard_predict():
    if request.method == "POST":
        input = request.form
        feature = [
            float(input['Bank_Accnt_Open']),
            float(input['Household_Size']),
            float(input['Homes_Owned']),
            float(input['Credit_Rating']),
            float(input['Average_Balance']),
            float(input['Q1_Balance']),
            float(input['Q2_Balance']),
            float(input['Q3_Balance']),
            float(input['Q4_Balance']),
            float(input['Reward_Air Miles']),
            float(input['Reward_Cash Back']),
            float(input['Reward_Points']),
            float(input['mailer_Letter']),
            float(input['mailer_Postcard']),
            float(input['income_High']),
            float(input['income_Low']),
            float(input['income_Medium']),
            float(input['overdraw_No']),
            float(input['overdraw_Yes']),
            float(input['CC_1']),
            float(input['CC_2']),
            float(input['CC_3']),
            float(input['CC_4']),
            float(input['hold_home_1']),
            float(input['hold_home_2']),
            float(input['hold_home_3'])
        ]
        
        pred= dst_ov.predict([feature])[0]
        pred_proba = dst_ov.predict_proba([feature])
        endresult = f"{round(np.max(pred_proba)*100,2)}% {'Accept the Offer' if pred == 1 else 'NOT Accept'}"


        # pred = gbc.predict([feature])[0]
        # pred_proba = gbc.predict_proba([feature])
        # pred_and_proba = f"{round(np.max(pred_proba)*100,2)}% {'BENIGN' if pred == 1 else 'NOT BENIGN'}"

        return render_template('result.html',
        data=input, prediction=endresult, worstconcave=input['worstconcave'],
        worstperim=input['worstperim'], meanconcave=input['meanconcave'],
        worstradius=input['worstradius'], meanperim=input['meanperim'],
        worstarea=input['worstarea'], meanradius=input['meanradius'],
        meanarea=input['meanarea'], meanconcavity=input['meanconcavity'],
        worstconcavity=input['worstconcavity'])

if __name__ == '__main__':
    dst_ov = joblib.load("DecisionTree-with-Oversampling") # load predictor

    app.run(debug=True, port=4000)
