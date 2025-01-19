from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import numpy as np
import wave

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = os.path.abspath('static/uploads')
DATA_FOLDER = os.path.abspath('data')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Fake database
users_db = {}

# User class
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    return users_db.get(user_id)

def save_user_data(username, features, audio_path):
    """Save user's voice sample and features to the local directory."""
    user_folder = os.path.join(DATA_FOLDER, username)
    os.makedirs(user_folder, exist_ok=True)
    print(f"[INFO] Saving features for user {username} in folder: {user_folder}")

    # Save features
    features_path = os.path.join(user_folder, 'voice_features.npy')
    np.save(features_path, features)
    print(f"[INFO] Features saved to {features_path}")

    # Save voice sample
    audio_file_path = os.path.join(user_folder, 'voice_sample.wav')
    os.replace(audio_path, audio_file_path)
    print(f"[INFO] Audio saved to {audio_file_path}")

def load_user_data(username):
    """Load user's voice features and sample path."""
    user_folder = os.path.join(DATA_FOLDER, username)
    if not os.path.exists(user_folder):
        print(f"[WARNING] User folder not found: {user_folder}")
        return None, None
    features = np.load(os.path.join(user_folder, 'voice_features.npy'))
    audio_file_path = os.path.join(user_folder, 'voice_sample.wav')
    return features, audio_file_path

def extract_features_from_audio(audio_path):
    """Extract features (mean and standard deviation) from a WAV file."""
    if not os.path.exists(audio_path):
        print(f"[ERROR] File not found: {audio_path}")
        raise FileNotFoundError(f"File not found: {audio_path}")

    # Otwórz plik WAV i wyciągnij cechy
    with wave.open(audio_path, 'rb') as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16)
    mean = np.mean(samples)
    std = np.std(samples)
    print(f"[INFO] Extracted features - Mean: {mean}, Std: {std}")
    return np.array([mean, std])

@app.route('/')
@login_required
def home():
    return render_template('panel.html', username=current_user.username)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        if not username:
            flash('Proszę podać nazwę użytkownika.')
            return redirect(url_for('register'))

        user_folder = os.path.join(DATA_FOLDER, username)
        if os.path.exists(user_folder):
            flash('Użytkownik już istnieje.')
            return redirect(url_for('register'))

        audio_file = request.files.get('audio')
        if not audio_file:
            flash('Proszę nagrać głos.')
            return redirect(url_for('register'))

        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.wav')
        audio_file.save(audio_path)
        print(f"[INFO] Audio file saved to {audio_path}")

        features = extract_features_from_audio(audio_path)
        user_id = str(len(users_db) + 1)
        new_user = User(user_id, username)
        users_db[user_id] = new_user
        save_user_data(username, features, audio_path)

        flash('Zarejestrowano pomyślnie!')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        if not username:
            return render_template('login.html', message='Proszę podać nazwę użytkownika.')

        user = next((u for u in users_db.values() if u.username == username), None)
        if not user:
            return render_template('login.html', message='Użytkownik nie istnieje.')

        audio_file = request.files.get('audio')
        if not audio_file:
            return render_template('login.html', message='Proszę nagrać głos.')

        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.wav')
        audio_file.save(audio_path)
        print(f"[INFO] Audio file saved to {audio_path}")

        features = extract_features_from_audio(audio_path)
        stored_features, _ = load_user_data(username)

        if stored_features is None:
            return render_template('login.html', message='Dane użytkownika nie zostały znalezione.')

        if np.linalg.norm(features - stored_features) < 5000.0:  # Threshold for voice verification
            login_user(user)
            return redirect(url_for('home'))
        else:
            return render_template('login.html', message='Weryfikacja głosu nieudana.')

    return render_template('login.html', message=None)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Wylogowano pomyślnie!')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
