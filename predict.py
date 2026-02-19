import os
import numpy as np
import pandas as pd
import librosa
import keras
from keras.models import model_from_json
import sys

def predict_emotion(audio_path):
    if not os.path.exists(audio_path):
        print(f"File {audio_path} not found.")
        return

    # Load the model
    print("Loading model...")
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # Load weights
    loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
    print("Model loaded.")

    # Feature extraction
    print(f"Extracting features from {audio_path}...")
    X, sample_rate = librosa.load(audio_path, res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    
    # Pad or truncate to 216 features (expected by model)
    if len(mfccs) > 216:
        mfccs = mfccs[:216]
    else:
        mfccs = np.pad(mfccs, (0, max(0, 216 - len(mfccs))), 'constant')

    livedf2 = pd.DataFrame(data=mfccs).stack().to_frame().T
    twodim = np.expand_dims(livedf2, axis=2)

    # Prediction
    print("Predicting...")
    livepreds = loaded_model.predict(twodim, batch_size=32, verbose=1)
    livepreds1 = livepreds.argmax(axis=1)

    # Mapping
    mapping = {
        0: 'female_angry',
        1: 'female_calm',
        2: 'female_fearful',
        3: 'female_happy',
        4: 'female_sad',
        5: 'male_angry',
        6: 'male_calm',
        7: 'male_fearful',
        8: 'male_happy',
        9: 'male_sad'
    }

    result = mapping.get(livepreds1[0], "Unknown")
    print(f"\nPredicted Emotion: {result}")
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Try a default file if it exists
        default_file = 'output10.wav'
        if os.path.exists(default_file):
            predict_emotion(default_file)
        else:
            print("Usage: python predict.py <path_to_audio_file>")
    else:
        predict_emotion(sys.argv[1])
