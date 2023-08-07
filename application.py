import pickle
from flask import Flask, request, jsonify, render_template, app, url_for
import numpy as np
import pandas as pd

application = Flask(__name__)
model = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@application.route('/')
def home():
    return render_template('home.html')

@application.route('/predict_api',methods=['POST'])
def predict_api():
    data= request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output= model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@application.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in  request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    prediction = model.predict(final_input)
    return render_template('home.html', prediction_text='Predicted Price is {}'.format(prediction[0]))

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=8000)