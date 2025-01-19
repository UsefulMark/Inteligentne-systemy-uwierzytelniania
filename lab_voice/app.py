import os
import json
import threading
import numpy as np
from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import bcrypt
import base64
import torchaudio
import torchaudio.transforms as T
from pydub import AudioSegment

# Konfiguracja aplikacji
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'

# Ścieżki do danych
USER_DATA_FOLDER = 'user_data'
USERS_FILE = os.path.join(USER_DATA_FOLDER, 'users.json')
AUDIO_TEMP_FILE = os.path.join(USER_DATA_FOLDER, 'temp_audio.wav')

# Blokada wątków
lock = threading.Lock()

# Inicjalizacja Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Sprawdzanie struktury folderów
os.makedirs(USER_DATA_FOLDER, exist_ok=True)
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'w') as f:
        json.dump({}, f)

# Klasa użytkownika
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

# Funkcja ładująca użytkownika
@login_manager.user_loader
def load_user(user_id):
    with lock:
        with open(USERS_FILE, 'r') as f:
            users = json.load(f)
    user_data = users.get(user_id)
    if user_data:
        return User(user_id, user_data['username'])
    return None

# Funkcja do zapisu audio z Base64 do pliku .wav z konwersją do poprawnego formatu
def save_audio_from_base64(base64_audio, output_path):
    try:
        audio_data = base64.b64decode(base64_audio)
        temp_path = output_path.replace(".wav", "_raw.wav")

        # Zapisz pierwotne audio do pliku tymczasowego
        with open(temp_path, 'wb') as f:
            f.write(audio_data)

        # Konwersja na poprawny format WAV
        audio = AudioSegment.from_file(temp_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")

        # Usuń plik tymczasowy
        os.remove(temp_path)
    except Exception as e:
        print(f"Błąd podczas zapisu audio: {e}")
        raise

# Funkcja do ekstrakcji cech MFCC z audio za pomocą torchaudio
def extract_mfcc(file_path):
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        mfcc_transform = T.MFCC(sample_rate=sample_rate, n_mfcc=13)
        mfcc = mfcc_transform(waveform)
        return mfcc.mean(dim=2).squeeze().tolist()
    except Exception as e:
        print(f"Błąd podczas ekstrakcji MFCC: {e}")
        return None

# Funkcja do porównywania cech za pomocą odległości kosinusowej
def match_voice_features(new_features, stored_features, threshold=0.4):
    distances = [np.dot(new_features, feature) / (np.linalg.norm(new_features) * np.linalg.norm(feature)) for feature in stored_features]
    return any(similarity >= threshold for similarity in distances)

@app.route('/')
@login_required
def home():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=current_user.username)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        base64_audio = request.form.get('voice')

        if not base64_audio:
            flash("Nie dostarczono nagrania głosu.")
            return redirect(url_for('register'))

        with lock:
            with open(USERS_FILE, 'r') as f:
                users = json.load(f)

            if username in [u['username'] for u in users.values()]:
                flash('Użytkownik już istnieje.')
                return redirect(url_for('register'))

            # Zapis audio i ekstrakcja cech
            save_audio_from_base64(base64_audio, AUDIO_TEMP_FILE)
            voice_features = extract_mfcc(AUDIO_TEMP_FILE)

            if not voice_features:
                flash("Nie udało się przetworzyć głosu.")
                return redirect(url_for('register'))

            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            user_id = str(len(users) + 1)
            users[user_id] = {
                'username': username,
                'password': hashed_password,
                'voice_features': [voice_features]
            }

            with open(USERS_FILE, 'w') as f:
                json.dump(users, f)

        flash('Zarejestrowano pomyślnie!')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        base64_audio = request.form.get('voice')

        if not base64_audio:
            flash("Nie dostarczono nagrania głosu.")
            return redirect(url_for('login'))

        with lock:
            with open(USERS_FILE, 'r') as f:
                users = json.load(f)

        user = next((u for u in users.values() if u['username'] == username), None)
        if not user or not bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            flash('Błędny login lub hasło.')
            return redirect(url_for('login'))

        # Zapis audio i ekstrakcja cech
        save_audio_from_base64(base64_audio, AUDIO_TEMP_FILE)
        voice_features = extract_mfcc(AUDIO_TEMP_FILE)

        if not voice_features:
            flash("Nie udało się przetworzyć głosu.")
            return redirect(url_for('login'))

        if match_voice_features(voice_features, user['voice_features']):
            user_id = next(key for key, value in users.items() if value['username'] == username)
            login_user(User(user_id, username))
            flash('Zalogowano pomyślnie!')
            return redirect(url_for('dashboard'))
        else:
            flash('Weryfikacja głosu nieudana.')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Wylogowano pomyślnie!')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
