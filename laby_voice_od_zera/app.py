from flask import Flask, request, render_template, jsonify
import librosa
import numpy as np
from pydub import AudioSegment
import io
import tempfile
import os

app = Flask(__name__)

# Funkcja do ekstrakcji cech audio
def extract_features(audio_data, sr):
    try:
        # Ekstrakcja współczynników MFCC (Mel-Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean.tolist()
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Strona główna aplikacji
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint do przetwarzania nagrania audio
@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    try:
        # Wczytanie pliku `webm` za pomocą `pydub`
        audio_data = AudioSegment.from_file(io.BytesIO(audio_file.read()), format="webm")

        # Konwersja `webm` na `wav`
        temp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio_data.export(temp_wav_file.name, format="wav")

        # Wczytanie pliku WAV za pomocą librosa
        y, sr = librosa.load(temp_wav_file.name, sr=None)

        # Ekstrakcja cech audio
        features = extract_features(y, sr)

        # Usuń tymczasowy plik
        os.unlink(temp_wav_file.name)

        if features is None:
            return jsonify({"error": "Failed to extract audio features"}), 500

        return jsonify({"features": features})
    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
