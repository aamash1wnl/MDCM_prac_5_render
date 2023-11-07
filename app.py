from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

app = Flask(__name__)

def test_transform(X_test):
    print(X_test.columns)
    X_test.drop(['Year','Model','Location','Make','Color','Fuel Type','Seller Type'], axis=1,inplace=True)
    print('niggameh')
    X_test['Owner'] = X_test['Owner'].map({'First': 4, 'Second': 3, 'UnRegistered Car': 2, 'Third': 1})
    X_test['Drivetrain'] = X_test['Drivetrain'].map({'FWD': 3, 'AWD': 4, 'RWD': 5})
    X_test['Transmission'] = X_test['Transmission'].map({'Automatic': 2, 'Manual': 3})
    print('niggaeheh1')
    # print(pd.DataFrame(pd.Series(X_test['Max Power'])))
    power_values = X_test['Max Power'].str.extract(r'(\d+) bhp')
    rpm_values = X_test['Max Power'].str.extract(r'@ (\d+) rpm')
    print('niggaeheh2')
    
    power_values1 = []
    rpm_values1 = []
    print('niggaeheh3')
    
    X_test['Power'] = X_test['Max Power'].str.extract(r'(\d+) bhp')
    X_test['RPM'] = X_test['Max Torque'].str.extract(r'@ (\d+) rpm')
    print('niggaeheh5')

    for i, j in enumerate(power_values):
        if type(power_values[i]) != float:
            power_values1.append(float(power_values[i]))
        elif type(power_values[i]) == float:
            power_values1.append(0)
    print('niggaeheh6')

    for i, j in enumerate(rpm_values):
        if type(rpm_values[i]) != float:
            rpm_values1.append(float(rpm_values[i]))
        elif type(rpm_values[i]) == float:
            rpm_values1.append(0)
    print('niggaeheh7')

    m_power = [power_values1[i] + rpm_values1[i] for i, j in enumerate(power_values1)]
    print('niggaeheh8')

    # Creating a new feature 'Max Torque' from two separate numerical values in a single feature
    X_test['C1'] = X_test['Max Torque'].str.extract(r'(\d+) Nm')
    X_test['C2'] = X_test['Max Power'].str.extract(r'@ (\d+) rpm')
    print('niggaeheh9')

    c1 = power_values
    c2 = rpm_values

    c11 = []
    c12 = []
    print('niggaeheh10')

    for i, j in enumerate(c1):
        if type(c1[i]) != float:
            c11.append(float(c1[i]))
        elif type(c1[i]) == float:
            c11.append(0)

    for i, j in enumerate(c2):
        if type(c2[i]) != float:
            c12.append(float(c2[i]))
        elif type(c2[i]) == float:
            c12.append(0)
    print('niggaeheh11')

    n_power = [c11[i] + c12[i] for i, j in enumerate(c11)]
    X_test['Max Power'] = m_power[0]
    X_test['Max Torque'] = n_power[0]
    print('niggaeheh12')
   
    print(X_test['Owner'].values[0])
    X_test['Transmission'] = X_test['Transmission'].values[0]
    X_test['Owner'] = X_test['Owner'].values[0]
    X_test['C1'] = X_test['C1'].values[0]
    X_test['C2'] = X_test['C2'].values[0]
    X_test['Drivetrain'] = X_test['Drivetrain'].values[0]
    X_test['Power'] = X_test['Power'].values[0]
    X_test['RPM'] = X_test['RPM'].values[0]
    print('niggaeheh13')

    X_test['Engine'] = X_test['Engine'].str.extract(r'(\d+)').astype(int)
    X_test = X_test.fillna(0)
    print('nigga14')
    X_test = pd.DataFrame(X_test)
    X_test = X_test[['Kilometer', 'Transmission', 'Owner', 'Engine', 'Max Power', 'Max Torque', 'Drivetrain','Length', 'Width', 'Height', 'Seating Capacity', 'Fuel Tank Capacity', 'Power', 'RPM', 'C1','C2']]
    return X_test

# Load the model, preprocessor, and scaler
pipeline = joblib.load('pipeline_prac_4_render.pkl')

column_order = [
    'Make', 'Model', 'Year', 'Kilometer', 'Fuel Type', 'Transmission', 'Location',
    'Color', 'Owner', 'Seller Type', 'Engine', 'Max Power', 'Max Torque',
    'Drivetrain', 'Length', 'Width', 'Height', 'Seating Capacity', 'Fuel Tank Capacity'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        # Get user input from the HTML form
        input_features = {col: request.form[col] for col in column_order}
        print(input_features)
        print(type(input_features))

        # Create a DataFrame from the user input
        input_df = pd.DataFrame(input_features, index=[0])
        input_df = test_transform(input_df)

        prediction = pipeline.predict(input_df)

        return render_template('index.html', prediction=f"Predicted Price: {np.abs(prediction[0])} USD")
    except Exception as e:
        error_message = str(e)
        print(f"An error occurred: {error_message}")
        return render_template('index.html', error=f"An error occurred: {error_message}")

if __name__ == '__main__':
    app.debug = True  # Enable debugging
    app.run()