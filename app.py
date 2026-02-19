import os
import warnings
# Silence all warnings for a clean project execution
warnings.filter_spec = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import pandas as pd
import librosa
import keras
from keras.models import model_from_json
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import traceback
from moviepy.editor import AudioFileClip
import speech_recognition as sr

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'm4a', 'mp4', 'mov'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Cache the model and graph
loaded_model = None
graph = None

def load_emotion_model():
    global loaded_model, graph
    if loaded_model is None:
        try:
            print("--- SENSE AI Core Initialization ---")
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            
            graph = tf.get_default_graph()
            
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
            print("AI Engine: Model Online")
        except Exception as e:
            print(f"FAILED TO LOAD MODEL: {e}")
            traceback.print_exc()
    return loaded_model, graph

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def convert_to_wav(input_path):
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{name}_converted.wav")
    
    audio = AudioFileClip(input_path)
    audio.write_audiofile(output_path, fps=22050, nbytes=2, codec='pcm_s16le', verbose=False, logger=None)
    audio.close()
    return output_path

def get_transcript(audio_path):
    """Simple speech to text transcription."""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        print(f"Transcription error: {e}")
        return "Could not transcribe audio."

def predict_emotion_logic(audio_path):
    model, current_graph = load_emotion_model()
    
    # Feature extraction
    X, sample_rate = librosa.load(audio_path, duration=2.5, sr=22050*2, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    
    if len(mfccs) > 216:
        mfccs = mfccs[:216]
    else:
        mfccs = np.pad(mfccs, (0, max(0, 216 - len(mfccs))), 'constant')

    livedf2 = pd.DataFrame(data=mfccs).stack().to_frame().T
    twodim = np.expand_dims(livedf2, axis=2)

    mapping = {
        0: 'Female Angry', 1: 'Female Calm', 2: 'Female Fearful', 3: 'Female Happy', 4: 'Female Sad',
        5: 'Male Angry', 6: 'Male Calm', 7: 'Male Fearful', 8: 'Male Happy', 9: 'Male Sad'
    }

    with current_graph.as_default():
        livepreds = model.predict(twodim, batch_size=32, verbose=0)
        
        probs = livepreds[0].tolist()
        formatted_probs = []
        for i, p in enumerate(probs):
            formatted_probs.append({
                'label': mapping[i],
                'value': round(float(p) * 100, 2)
            })
        
        top_emotions = sorted(formatted_probs, key=lambda x: x['value'], reverse=True)[:3]
        livepreds1 = livepreds.argmax(axis=1)

    res = mapping.get(livepreds1[0], "Unknown")
    insights = get_emotion_insights(res)
    
    return res, insights, top_emotions

def get_emotion_insights(emotion):
    insights_map = {
        'Female Angry': "High intensity signals detected. The tone suggests frustration or strong dissatisfaction.",
        'Female Calm': "Composed and neutral tone. Signals reliability and logical processing.",
        'Female Fearful': "Vocal jitter detected. Tone indicates high stress or situational apprehension.",
        'Female Happy': "Positive vocal energy. Indicates strong rapport and satisfaction.",
        'Female Sad': "Low intensity response. Signals displacement, fatigue, or low engagement.",
        'Male Angry': "Dominant vocal push. Indicates assertive confrontation or high frustration.",
        'Male Calm': "Stable and resonant frequency. Signals authority and professional reliability.",
        'Male Fearful': "Rising vocal pitch. Indicates uncertainty or situational anxiety.",
        'Male Happy': "Uplifting vocal patterns. High indicators of enthusiasm and positive bonding.",
        'Male Sad': "Withdrawn vocal resonances. Indicates somber mood or low motivation."
    }
    return insights_map.get(emotion, "General vocal patterns detected.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        origin_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(origin_path)
        
        processing_path = origin_path
        needs_cleanup = [origin_path]
        
        try:
            if not filename.lower().endswith('.wav'):
                processing_path = convert_to_wav(origin_path)
                needs_cleanup.append(processing_path)
            
            emotion, insights, probabilities = predict_emotion_logic(processing_path)
            
            # Transcription (Scientific Value)
            transcript = get_transcript(processing_path)
            
            y, sr = librosa.load(processing_path)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            pitch_mean = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            energy = float(np.mean(librosa.feature.rms(y=y)))
            
            for p in needs_cleanup:
                if os.path.exists(p): os.remove(p)
                
            return jsonify({
                'emotion': emotion, 
                'insights': insights, 
                'probabilities': probabilities,
                'transcript': transcript,
                'metrics': {
                    'tempo': round(float(tempo), 1),
                    'pitch': round(pitch_mean, 0),
                    'energy': round(energy * 100, 2)
                }
            })
        except Exception as e:
            traceback.print_exc()
            for p in needs_cleanup:
                if os.path.exists(p): os.remove(p)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Unsupported file format'}), 400

if __name__ == '__main__':
    load_emotion_model()
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=False)
