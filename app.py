import pickle
from flask import Flask, request, jsonify, render_template, app, url_for
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data= request.json['data']
    print(data)