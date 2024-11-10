from flask import Flask, request, render_template, redirect, url_for, session
import os
import cv2
import numpy as np
import torch

# Ładowanie wytrenowanego modelu
class CombinedFingerprintModel(torch.nn.Module):
    def __init__(self, cnn_output_size=128, geometry_input_size=4, num_classes=10):
        super(CombinedFingerprintModel, self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 32 * 32, cnn_output_size),
            torch.nn.ReLU()
        )
        self.combined_fc = torch.nn.Sequential(
            torch.nn.Linear(cnn_output_size + geometry_input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )
    
    def forward(self, image, geometry_features):
        cnn_features = self.cnn(image)
        combined_features = torch.cat((cnn_features, geometry_features), dim=1)
        output = self.combined_fc(combined_features)
        return output

model = CombinedFingerprintModel()
model.load_state_dict(torch.load("combined_fingerprint_model.pth"))
model.eval()

# Funkcja do ekstrakcji cech geometrycznych
def extract_geometric_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Nie znaleziono obrazu: {image_path}")
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("Nie znaleziono konturów na obrazie.")
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / h
    extent = area / (w * h)
    return np.array([area, perimeter, aspect_ratio, extent])

# Flask app
app = Flask(__name__, template_folder="templates2")
app.secret_key = "supersecretkey"
UPLOAD_FOLDER = "uploads"
DATABASE_FOLDER = "database"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATABASE_FOLDER, exist_ok=True)

@app.route("/")
def home():
    username = session.get("username")
    return render_template("index.html", username=username)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        file = request.files["fingerprint"]
        
        if not file:
            return "Nie przesłano pliku", 400
        
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        # Ekstrakcja cech CNN
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 128)) / 255.0
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            cnn_features = model.cnn(image_tensor).flatten().numpy()
        np.savetxt(os.path.join(DATABASE_FOLDER, f"{username}_cnn.txt"), cnn_features, fmt="%.5f")
        
        # Ekstrakcja cech geometrycznych
        geometry_features = extract_geometric_features(file_path)
        np.savetxt(os.path.join(DATABASE_FOLDER, f"{username}_geometry.txt"), geometry_features, fmt="%.5f")
        
        return render_template("register_success.html", username=username, geometry_features=geometry_features)
    
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        file = request.files["fingerprint"]
        
        if not file:
            return "Nie przesłano pliku", 400
        
        cnn_file = os.path.join(DATABASE_FOLDER, f"{username}_cnn.txt")
        geometry_file = os.path.join(DATABASE_FOLDER, f"{username}_geometry.txt")
        
        if not os.path.exists(cnn_file) or not os.path.exists(geometry_file):
            return "Użytkownik nie istnieje", 404
        
        registered_cnn_features = np.loadtxt(cnn_file)
        registered_geometry_features = np.loadtxt(geometry_file)
        
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        # Ekstrakcja nowych cech CNN
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 128)) / 255.0
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            new_cnn_features = model.cnn(image_tensor).flatten().numpy()
        
        # Ekstrakcja nowych cech geometrycznych
        new_geometry_features = extract_geometric_features(file_path)
        
        cnn_distance = np.linalg.norm(registered_cnn_features - new_cnn_features)
        geometry_distance = np.linalg.norm(registered_geometry_features - new_geometry_features)
        
        session["username"] = username
        if cnn_distance < 10.0 and geometry_distance < 5.0:
            return render_template("login_success.html", username=username, cnn_distance=cnn_distance, geometry_distance=geometry_distance)
        else:
            return render_template("login_failure.html", cnn_distance=cnn_distance, geometry_distance=geometry_distance)
    
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
