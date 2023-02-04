import numpy as np
import os
from keras.preprocessing import image 
import pandas as pd
import cv2
import tensorflow as tf
# Flask utils
from flask import Flask,request, render_template
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

global graph
tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
graph=tf.get_default_graph()

# Define a flask app
app = Flask(__name__)
set_session(sess)
# Load your trained model
model = load_model('adp.h5')

print('Model loaded. Check http://127.0.0.1:5000/')


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('alzheimers.html')

@app.route('/predict1', methods=['GET'])
def predict1():
    # Main page
    return render_template('alzpre.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']
        
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        with graph.as_default():
            set_session(sess)
            prediction = model.predict(x)[0][0][0]
        print(prediction)
        if prediction==0:
            text = "Mild Demented"
        elif prediction==1:
                text = "Moderate Demented"
        elif prediction==2:
                text = "Non Demented"
        else:
            text = "Very Mild Demented"

        return text

if __name__ == '__main__':
    app.run(debug=False,threaded = False)
