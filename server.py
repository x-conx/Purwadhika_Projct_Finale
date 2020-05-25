import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, url_for,send_from_directory
import numpy as np

app = Flask(__name__)
@app.route('/Home-CreditCard-Prediction')
def home():
    return render_template('Home.html')

@app.route('/gallery/<path:x>')
def gal(x):
    return send_from_directory("gallery",x)

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
        rewards = input["reward_bonus"]
        strRew = ''
        if rewards == "Reward_Air_Miles":
            Reward_Air_Miles = 1
            Reward_Cash_Back = 0
            Reward_Points = 0
            strRew = 'Air Miles'
        if rewards == "Reward_Cash_Back":
            Reward_Air_Miles = 0
            Reward_Cash_Back = 1
            Reward_Points = 0
            strRew = 'Cash Back'
        if rewards == "Reward_Points":
            Reward_Air_Miles = 0
            Reward_Cash_Back = 0
            Reward_Points = 1
            strRew = 'Credit Card Point'
            
        # mailer_Letter = int(input['mailer_Letter'])
        # mailer_Postcard = int(input['mailer_Postcard'])
        mails = input['mailers']
        strmail = ''
        if mails == 'letter':
            mailer_Letter = 1
            mailer_Postcard = 0
            strmail = 'Letter'
        else:
            mailer_Letter = 0
            mailer_Postcard = 1
            strmail = 'Postcard'

        # income_High = int(input['income_High'])
        # income_Low = int(input['income_Low'])
        # income_Medium = int(input['income_Medium'])
        inm = input['income']
        strin = ''
        if inm == 'high':
            income_High = 1
            income_Medium = 0
            income_Low = 0
            strin = 'High'
        elif inm == 'medium':
            income_High = 0
            income_Medium = 1
            income_Low = 0
            strin = 'Medium'
        else:
            income_High = 0
            income_Medium = 0
            income_Low = 1
            strin = 'Low'
        
        # overdraw_No = int(input['overdraw_No'])
        # overdraw_Yes = int(input['overdraw_Yes'])
        prot = input['ovprot']
        strovrd = ''
        if prot == 'yes':
            overdraw_No = 0
            overdraw_Yes = 1
            strovrd = 'Yes'
        else:
            overdraw_No = 1
            overdraw_Yes = 0
            strovrd = 'No'

        # CC_1 = int(input['CC_1'])
        # CC_2 = int(input['CC_2'])
        # CC_3 = int(input['CC_3'])
        # CC_4 = int(input['CC_4'])
        ccx = input['cc']
        strcc = ''
        if ccx == '1':
            CC_1 = 1
            CC_2 = 0
            CC_3 = 0
            CC_4 = 0
            strcc = '1'
        if ccx == "2":
            CC_1 = 0
            CC_2 = 1
            CC_3 = 0
            CC_4 = 0
            strcc = '2'
        if ccx == '3':
            CC_1 = 0
            CC_2 = 0
            CC_3 = 1
            CC_4 = 0
            strcc = '3'
        if ccx == '4':
            CC_1 = 0
            CC_2 = 0
            CC_3 = 0
            CC_4 = 1
            strcc = '4'

        # hold_home_1 = int(input['hold_home_1'])
        # hold_home_2 = int(input['hold_home_2'])
        # hold_home_3 = int(input['hold_home_3'])
        homes = input['hom']
        strhom = ''
        if homes == '1':
            hold_home_1 = 1
            hold_home_2 = 0
            hold_home_3 = 0
            strhom = '1'
        if homes == '2':
            hold_home_1 = 0
            hold_home_2 = 1
            hold_home_3 = 0
            strhom = '2'
        if homes == '3':
            hold_home_1 = 0
            hold_home_2 = 0
            hold_home_3 = 1
            strhom = '3'

        Bank_Accnt_Open = int(input['Bank_Accnt_Open'])
        Household_Size = int(input['Household_Size'])
        Homes_Owned = int(input['Homes_Owned'])
        Credit_Rating = int(input['Credit_Rating'])
        Average_Balance = float(input['Average_Balance'])
        Q1_Balance = float(input['Q1_Balance'])
        Q2_Balance = float(input['Q2_Balance'])
        Q3_Balance = float(input['Q3_Balance'])
        Q4_Balance = float(input['Q4_Balance'])
        
        feature = [Bank_Accnt_Open, Household_Size, Homes_Owned,Credit_Rating, Average_Balance, Q1_Balance, Q2_Balance,
        Q3_Balance, Q4_Balance, Reward_Air_Miles, Reward_Cash_Back,Reward_Points, mailer_Letter, mailer_Postcard, income_High,
        income_Low, income_Medium, overdraw_No, overdraw_Yes, CC_1,CC_2, CC_3, CC_4, hold_home_1, hold_home_2, hold_home_3]

        pred= dst_ov.predict([feature])
        pred_proba = dst_ov.predict_proba([feature])
        if pred == 0:
            endresult = f"{round(pred_proba.max()*100)}% {'NOT Accept the Offer'}"
            col = 'red'
        else:
            endresult = f"{round(pred_proba.max()*100)}% {'Customer will Accept the Offer'}"
            col = 'green'


        
        return render_template('prediction result.html',
        data=input, prediction=endresult, Bank_Accnt_Open=Bank_Accnt_Open,Household_Size=Household_Size,
        Homes_Owned=Homes_Owned,Credit_Rating=Credit_Rating,Average_Balance=Average_Balance,
        Q1_Balance=Q1_Balance,Q2_Balance=Q2_Balance,Q3_Balance=Q3_Balance,
        Q4_Balance=Q4_Balance,Reward=strRew,mailer=strmail,
        income=strin,overdraw=strovrd,cc=strcc,hold_home=strhom,color=col)

if __name__ == '__main__':
    dst_ov = joblib.load("DecisionTree-with-Oversampling") # load predictor

    app.run(debug=True, port=4000)
