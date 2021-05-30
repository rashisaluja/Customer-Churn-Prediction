import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

df_1 = pd.read_csv("Dataset/first_telc.csv")
q = ''
@app.route('/')
def loadPage():
	return render_template('index.html', query="")

@app.route('/',methods=['POST'])
def predict():

    if request.method == 'POST':
        SeniorCitizen = request.form['SeniorCitizen']
        MonthlyCharges = request.form['MonthlyCharges']
        TotalCharges = request.form['TotalCharges']
        Gender = request.form['Gender']
        Partner = request.form['Partner']
        Dependents = request.form['Dependents']
        PhoneService = request.form['PhoneService']
        MultipleLines = request.form['MultipleLines']
        InternetService = request.form['InternetService']
        OnlineSecurity = request.form['OnlineSecurity']
        OnlineBackup = request.form['OnlineBackup']
        DeviceProtection = request.form['DeviceProtection']
        TechSupport = request.form['TechSupport']
        StreamingTV = request.form['StreamingTV']
        StreamingMovies = request.form['StreamingMovies']
        Contract = request.form['Contract']
        PaperlessBilling = request.form['PaperlessBilling']
        PaymentMethod = request.form['PaymentMethod']
        Tenure = request.form['Tenure']

        data = [[SeniorCitizen, MonthlyCharges, TotalCharges, Gender, Partner, Dependents, PhoneService, MultipleLines, InternetService,
                OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract,
                PaperlessBilling, PaymentMethod, Tenure]]

        new_df = pd.DataFrame(data, columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'Gender',
                                             'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                             'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                             'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                             'PaymentMethod', 'Tenure'])
        df_2 = pd.concat([df_1, new_df], ignore_index=True)
        # Group the tenure in bins of 12 months
        labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

        df_2['tenure_group'] = pd.cut(df_2.Tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
        # drop column customerID and tenure
        df_2.drop(columns=['Tenure'], axis=1, inplace=True)

        new_df__dummies = pd.get_dummies(df_2[['Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                               'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                               'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                               'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']])

        # final_df=pd.concat([new_df__dummies, new_dummy], axis=1)

        single = model.predict(new_df__dummies.tail(1))
        probablity = model.predict_proba(new_df__dummies.tail(1))[:, 1]

        if single == 1:
            o1 = "This customer is likely to be churned!!"
            o2 = "Confidence: {}".format(probablity * 100)
        else:
            o1 = "This customer is likely to continue!!"
            o2 = "Confidence: {}".format(probablity * 100)

        return render_template('index.html', prediction_text = '{}'.format(o1), predicted_confidence='{}'.format(o2),
            seniorCitizen = request.form['SeniorCitizen'],
            monthlyCharges = request.form['MonthlyCharges'],
            totalCharges = request.form['TotalCharges'],
            gender = request.form['Gender'],
            dependents = request.form['Dependents'],
            phoneService = request.form['PhoneService'],
            multipleLines = request.form['MultipleLines'],
            internetService = request.form['InternetService'],
            onlineSecurity = request.form['OnlineSecurity'],
            onlineBackup = request.form['OnlineBackup'],
            deviceProtection = request.form['DeviceProtection'],
            techSupport = request.form['TechSupport'],
            streamingTV = request.form['StreamingTV'],
            streamingMovies = request.form['StreamingMovies'],
            contract = request.form['Contract'],
            paperlessBilling = request.form['PaperlessBilling'],
            paymentMethod = request.form['PaymentMethod'],
            tenure = request.form['Tenure'])





if __name__ == "__main__":
    app.run(debug=True)