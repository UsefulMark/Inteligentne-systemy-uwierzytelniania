from flask import Flask, request, render_template, redirect, url_for, session
import os
import cv2
import numpy as np
import torch

# Ładowanie wytrenowanego modelu
class FingerprintCNN(torch.nn.Module):
    def __init__(self):
        super(FingerprintCNN, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 32 * 32, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)  # Zakładamy 10 klas
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model = FingerprintCNN()
model.load_state_dict(torch.load("fingerprint_cnn.pth"))
model.eval()

# Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Klucz dla sesji
UPLOAD_FOLDER = "uploads"
DATABASE_FOLDER = "database"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATABASE_FOLDER, exist_ok=True)

@app.route("/")
def home():
    username = session.get("username")  # Pobierz użytkownika z sesji
    return render_template("index.html", username=username)



@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        file = request.files["fingerprint"]
        
        if not file:
            return "Nie przesłano pliku", 400
        
        # Zapisz obraz
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        # Przetwarzanie obrazu i zapis cech
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return "Nieprawidłowy plik obrazu", 400
        image = cv2.resize(image, (128, 128)) / 255.0
        image = np.expand_dims(image, axis=(0, 1))
        image_tensor = torch.tensor(image, dtype=torch.float32)
        
        with torch.no_grad():
            features = model.conv_layers(image_tensor).flatten().numpy()
        
        # Zapis cech do bazy
        user_file = os.path.join(DATABASE_FOLDER, f"{username}.txt")
        np.savetxt(user_file, features, fmt="%.5f")
        
        # Przekierowanie na stronę sukcesu (bez ustawiania sesji)
        return render_template("register_success.html", username=username)
    
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        file = request.files["fingerprint"]
        
        if not file:
            return "Nie przesłano pliku", 400
        
        # Ścieżka do pliku użytkownika
        user_file = os.path.join(DATABASE_FOLDER, f"{username}.txt")
        if not os.path.exists(user_file):
            return "Użytkownik nie istnieje", 404
        
        # Wczytaj zapisane cechy
        registered_features = np.loadtxt(user_file)
        
        # Przetwarzanie obrazu
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return "Nieprawidłowy plik obrazu", 400
        image = cv2.resize(image, (128, 128)) / 255.0
        image = np.expand_dims(image, axis=(0, 1))
        image_tensor = torch.tensor(image, dtype=torch.float32)
        
        with torch.no_grad():
            new_features = model.conv_layers(image_tensor).flatten().numpy()
        
        # Oblicz odległość euklidesową
        distance = np.linalg.norm(registered_features - new_features)
        session["username"] = username  # Zapisz użytkownika w sesji
        if distance < 10.0:  # Próg logowania
            return render_template("login_success.html", username=username, distance=distance)
        else:
            return render_template("login_failure.html", distance=distance)
    
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("username", None)  # Usuń użytkownika z sesji
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
