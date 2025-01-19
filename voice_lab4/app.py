from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import json
import speech_recognition as sr

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Fake database: przechowuje użytkowników (klucz to login)
users_db = {}
voice_features = {}

# Klasa User
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

# Funkcja ładująca użytkownika na podstawie ID
@login_manager.user_loader
def load_user(user_id):
    return users_db.get(user_id)

def extract_features_from_microphone():
    """Przechwytuje dźwięk z mikrofonu i ekstraktuje cechy."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Nagrywanie: proszę mówić przez kilka sekund...")
        try:
            audio = recognizer.listen(source, timeout=5)
            print("Nagranie zakończone. Przetwarzanie głosu...")
            features = recognizer.recognize_google(audio)
            print(f"Rozpoznane cechy: {features}")
            return features
        except sr.UnknownValueError:
            print("Nie udało się rozpoznać mowy.")
            return None
        except sr.RequestError as e:
            print(f"Błąd w połączeniu z usługą rozpoznawania: {e}")
            return None
        except Exception as e:
            print(f"Wystąpił błąd: {e}")
            return None

# Funkcja do zapisu danych do pliku
def save_data():
    with open('data.json', 'w') as f:
        json.dump({
            "users": {user_id: user.username for user_id, user in users_db.items()},
            "voice_features": voice_features
        }, f)

# Funkcja do wczytania danych z pliku
def load_data():
    if os.path.exists('data.json'):
        with open('data.json', 'r') as f:
            data = json.load(f)
            for user_id, username in data["users"].items():
                users_db[user_id] = User(user_id, username)
            global voice_features
            voice_features = data["voice_features"]

# Wczytanie danych przy starcie aplikacji
load_data()

@app.route('/')
@login_required
def home():
    return f'Witaj, {current_user.username}! <br><a href="{url_for("logout")}">Wyloguj</a>'

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        if username in [user.username for user in users_db.values()]:
            flash('Użytkownik już istnieje.')
            return redirect(url_for('register'))

        features = extract_features_from_microphone()
        if features is None:
            flash('Nie udało się przetworzyć głosu. Spróbuj ponownie.')
            return redirect(url_for('register'))

        user_id = str(len(users_db) + 1)
        new_user = User(user_id, username)
        users_db[user_id] = new_user

        if username not in voice_features:
            voice_features[username] = []
        voice_features[username].append(features)

        # Zapis danych po rejestracji
        save_data()

        flash('Zarejestrowano pomyślnie!')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')

        user = next((u for u in users_db.values() if u.username == username), None)
        if not user:
            flash('Użytkownik nie istnieje.')
            return redirect(url_for('login'))

        features = extract_features_from_microphone()
        if features is None:
            flash('Nie udało się przetworzyć głosu.')
            return redirect(url_for('login'))

        known_features = voice_features.get(username, [])

        if features in known_features:
            login_user(user)
            flash('Zalogowano pomyślnie!')
            return redirect(url_for('home'))
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
