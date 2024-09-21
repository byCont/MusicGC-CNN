from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
import librosa
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from pydub import AudioSegment

app = Flask(__name__)

# Constants (make sure these match the values used during training)
SAMPLE_RATE = 22050
DURATION = 10
NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
NUM_SEGMENTS = 10

# List of genres (make sure this matches your training data)
GENRES = ['bachata', 'reggaeton', 'merengue', 'salsa', 'popular']  # Update this list with your actual genres

# Load the model
model = load_model('latin_music_genre_classifier.h5')

# Initialize LabelEncoder
le = LabelEncoder()

# Check if label_encoder_classes.npy exists, if not create it
if not os.path.exists('label_encoder_classes.npy'):
    le.fit(GENRES)
    np.save('label_encoder_classes.npy', le.classes_)
else:
    le.classes_ = np.load('label_encoder_classes.npy')

def split_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    segment = audio[:DURATION * 1000]  # Convert DURATION to milliseconds
    return segment

def extract_features(audio_path):
    audio, _ = librosa.load(audio_path, duration=DURATION, sr=SAMPLE_RATE, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=NUM_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return mfccs.T

def predict_genre(audio_path):
    features = extract_features(audio_path)
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction, axis=1)
    predicted_genre = le.inverse_transform(predicted_index)[0]
    return predicted_genre

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/classify")
def classify():
    return render_template('classify.html')

@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST':
        audio_file = request.files['wavfile']
        filename = secure_filename(audio_file.filename)
        audio_path = os.path.join('static', 'tests', filename)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        
        audio_file.save(audio_path)

        # Split audio if necessary
        if AudioSegment.from_file(audio_path).duration_seconds > DURATION:
            split_audio(audio_path).export(audio_path, format="wav")

        predicted_genre = predict_genre(audio_path)

        return render_template("prediction.html", prediction=predicted_genre, audio_path=audio_path)

@app.route("/about")
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)