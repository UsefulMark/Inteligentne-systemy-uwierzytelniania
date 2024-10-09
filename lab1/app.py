#app.py

from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import face_recognition
import os
from werkzeug.utils import secure_filename
 
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey123'
app.config['UPLOAD_FOLDER'] = 'uploads'  # Katalog do przechowywania załadowanych obrazów
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maksymalny rozmiar pliku (16 MB)
 
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
 
# Fake database: przechowuje użytkowników (klucz to login)
users_db = {}
 
# Klasa User
class User(UserMixin):
    def __init__(self, id, username, face_encoding):
        self.id = id
        self.username = username
        self.face_encoding = face_encoding
 
# Funkcja ładująca użytkownika na podstawie ID
@login_manager.user_loader
def load_user(user_id):
    return users_db.get(user_id)
 
# Strona główna - wymaga zalogowania
@app.route('/')
@login_required
def home():
    return f'Witaj, {current_user.username}! <br><a href=''>Wyloguj</a>'
 
# Rejestracja nowego użytkownika
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        if username in [user.username for user in users_db.values()]:
            flash('Użytkownik już istnieje.')
            return redirect(url_for('register'))
 
        # Sprawdzenie, czy plik został załadowany
        if 'file' not in request.files:
            flash('Nie przesłano pliku.')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Nie wybrano pliku.')
            return redirect(request.url)
        
        # Zapis pliku
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
 
        # Zarejestruj twarz z przesłanego pliku
        face_encoding = get_face_encoding_from_image(file_path)
        if face_encoding is None:
            flash('Nie udało się rozpoznać twarzy w przesłanym obrazie.')
            return redirect(url_for('register'))
 
        user_id = str(len(users_db) + 1)
        new_user = User(user_id, username, face_encoding)
        users_db[user_id] = new_user
        flash('Zarejestrowano pomyślnie, teraz możesz się zalogować.')
        return redirect(url_for('login'))
    
    return render_template('register.html')
 
# Logowanie użytkownika za pomocą przesłanego obrazu
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
 
        user = next((u for u in users_db.values() if u.username == username), None)
        if not user:
            flash('Użytkownik nie istnieje.')
            return redirect(url_for('login'))
 
        # Sprawdzenie, czy plik został załadowany
        if 'file' not in request.files:
            flash('Nie przesłano pliku.')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Nie wybrano pliku.')
            return redirect(request.url)
        
        # Zapis pliku
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
 
        # Sprawdzenie twarzy z przesłanego pliku
        captured_face_encoding = get_face_encoding_from_image(file_path)
        if captured_face_encoding is None:
            flash('Nie udało się rozpoznać twarzy w przesłanym obrazie.')
            return redirect(url_for('login'))
 
        # Porównanie twarzy z zarejestrowaną
        results = face_recognition.compare_faces([user.face_encoding], captured_face_encoding)
        if results[0]:
            login_user(user)
            flash('Zalogowano pomyślnie!')
            return redirect(url_for('home'))
        else:
            flash('Rozpoznanie twarzy nieudane.')
            return redirect(url_for('login'))
    
    return render_template('login.html')
 
# Wylogowanie użytkownika
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Wylogowano pomyślnie!')
    return redirect(url_for('login'))
 
def get_face_encoding_from_image(image_path):
    """Pobierz encoding twarzy z przesłanego obrazu."""
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    if face_locations:
        face_encodings = face_recognition.face_encodings(image, face_locations)
        return face_encodings[0] if face_encodings else None
    return None
 
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
