

import pickle

import numpy as np
from flask import Flask, request, render_template

#loading models
dtr = pickle.load(open('dtr.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))

#creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        Year = request.form['Year']
        Cost_Cultivation_A2_FL =request.form['Cost_Cultivation_A2_FL']
        Cost_Cultivation_C2 =request.form['Cost_Cultivation_C2']
        Cost_Production_C2 =request.form['Cost_Production_C2']
        Cost_per_Hectare =request.form['Cost_per_Hectare ']
        Cost_per_Quintal =request.form['Cost_per_Quintal']

        features = np.array([[Area_2006_7,Area_2007_8,Area_2009_10,Area_2010_11,Cost_Cultivation_A2_FL,Cost_Cultivation_C2,Cost_Production_C2,Cost_per_Hectare,Cost_per_Quintal]])

        transfromed_features = preprocessor.transform(features)
        predicted_value = dtr.predict(transfromed_features).reshape(1,-1)

        return render_template('index.html',predicted_value=predicted_value)







#python main
if __name__=='__main__':
    app.run(debug=True)