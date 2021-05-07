from flask import Flask,render_template,request,Markup
import numpy as np
import os

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from werkzeug.utils import secure_filename
from recommend import crop_recommend
from soil_predict import model_predict
from plant_predict import disease_predict

import pandas as pd
from fertilizer import fertilizer_dic
import requests



app = Flask(__name__)

# load model
model = load_model('model_soil_classification.h5')
# load model
model_plant = load_model('model_plant_disease.h5')


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/fertilizer_recommend')
def fertilizer_recommennd():
    return render_template('fertilizer_recommend.html')

@app.route('/fert_recommend',methods=['POST','GET'])
def fert_recommend():
    
    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K

    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))
    return render_template('fertilizer-result.html', recommendation=response)


@app.route('/plant_disease')
def plant_disease():
    return render_template('plant_disease.html')

IMG_FOLDER = os.path.join('static', 'upload')

app.config['UPLOAD_FOLDER'] = IMG_FOLDER

@app.route('/predict_plant',methods=['GET','POST'])
def predict():
	if request.method == 'POST':
		file = request.files['image'] # fetch input
		filename = file.filename 

		file_path = os.path.join('static/upload', filename)
		file.save(file_path)

		pred = disease_predict(file_path,model_plant)

		full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)


		return render_template('plant_disease.html',predict = pred,user_image = full_filename)




@app.route('/soil')
def soil():
    return render_template('soil.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None



if __name__ == '__main__':
	app.run(debug=True)