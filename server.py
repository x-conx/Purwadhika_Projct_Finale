import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, url_for

app = Flask(__name__)
@app.route('/Home-CreditCard-Prediction')
def home():
    return render_template('Home.html')

@app.route('/About')
def about():
    return render_template('about.html')

@app.route('/Visualisation')
def vis():
    return render_template('visual.html')

@app.route('/')
def prediction():
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
            float(input['Reward_Air_Miles']),
            float(input['Reward_Cash_Back']),
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
        
        pred= dst_ov.predict([feature])
        pred_proba = dst_ov.predict_proba([feature])
        if pred == 0:
            endresult = f"{round(pred_proba.max()*100)}% {'NOT Accept the Offer'}"
        else:
            endresult = f"{round(pred_proba.max()*100)}% {'Customer will Accept the Offer'}"


        
        return render_template('prediction result.html',
        data=input, prediction=endresult, Bank_Accnt_Open=input['Bank_Accnt_Open'],Household_Size=input['Household_Size'],
        Homes_Owned=input['Homes_Owned'],Credit_Rating=input['Credit_Rating'],Average_Balance=input['Average_Balance'],
        Q1_Balance=input['Q1_Balance'],Q2_Balance=input['Q2_Balance'],Q3_Balance=input['Q3_Balance'],
        Q4_Balance=input['Q4_Balance'],Reward_Air_Miles=input['Reward_Air_Miles'],Reward_Cash_Back=input['Reward_Cash_Back'],
        Reward_Points=input['Reward_Points'],mailer_Letter=input['mailer_Letter'],mailer_Postcard=input['mailer_Postcard'],
        income_High=input['income_High'],income_Low=input['income_Low'],income_Medium=input['income_Medium'],
        overdraw_No=input['overdraw_No'],overdraw_Yes=input['overdraw_Yes'],CC_1=input['CC_1'],CC_2=input['CC_2'],
        CC_3=input['CC_3'],CC_4=input['CC_4'],hold_home_1=input['hold_home_1'],hold_home_2=input['hold_home_2'],
        hold_home_3=input['hold_home_3'])

if __name__ == '__main__':
    dst_ov = joblib.load("DecisionTree-with-Oversampling") # load predictor

    app.run(debug=True, port=4000)
