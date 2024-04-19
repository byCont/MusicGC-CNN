from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.models import load_model
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import pandas as pd
import os
import librosa
import librosa, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
 
from scipy.io.wavfile import write
 
# from flask import Flask, redirect, url_for, request, render_template
import numpy as np
import pandas as pd
import time
import os
from collections import Counter
from pydub import AudioSegment
### Flask imports
import requests
from flask import Flask, render_template, session, request, redirect, flash, Response
app = Flask(__name__)
 
 
final = pd.read_pickle("extracted_df.pkl")
y = np.array(final["label"].tolist())
le = LabelEncoder()
le.fit_transform(y)
model = tf.keras.models.load_model('Model1.h5', compile=False)


def split_audio(file_path):
    # Cargar el archivo de audio
    audio = AudioSegment.from_file(file_path)

    # Tomar los primeros 30 segundos del audio
    segment = audio[:30000]  # 30 segundos representados en milisegundos (30,000 ms)
		# output_file = f"{file_path}"
    return segment


def extract_feature(audio_path):
    audio_data, sample_rate = librosa.load(audio_path, res_type="kaiser_fast")
    feature = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    feature_scaled = np.mean(feature.T, axis=0)
    return np.array([feature_scaled])


def ANN_print_prediction(audio_path):
    prediction_feature = extract_feature(audio_path)
    predicted_vector = np.argmax(model.predict(prediction_feature), axis=-1)
    predicted_class = le.inverse_transform(predicted_vector)
    return predicted_class[0]

@app.route("/")
@app.route("/home")
def home():
	return render_template('home.html')
    
@app.route("/classify")
def classify():
	return render_template('classify.html')


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		audio_path = request.files['wavfile']

		img_path = "static/tests/" + audio_path.filename	
		audio_path.save(img_path)
	 
		predict_result =  ANN_print_prediction(img_path)

	return render_template("prediction.html", prediction = predict_result, audio_path= img_path)

@app.route("/about")
def about():
	return render_template('about.html')     
 
if __name__ =='__main__':
	app.run(debug = True)
