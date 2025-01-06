from flask import Flask, request, render_template, redirect, url_for, session, flash
import os
import cv2
import numpy as np
import torch
import hashlib

# Model
class CombinedFingerprintModel(torch.nn.Module):
    def __init__(self, cnn_output_size=128, geometry_input_size=4, minutiae_input_size=2, num_classes=10):
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
            torch.nn.Linear(cnn_output_size + geometry_input_size + minutiae_input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, image, geometry_features, minutiae_features):
        cnn_features = self.cnn(image)
        combined_features = torch.cat((cnn_features, geometry_features, minutiae_features), dim=1)
        output = self.combined_fc(combined_features)
        return output

# # Wczytanie modelu na GPU
# model = CombinedFingerprintModel()
# model.load_state_dict(torch.load("combined_fingerprint_model_with_minutiae.pth"))
# model.eval()

# Wczytanie modelu
model = CombinedFingerprintModel()

# Wymuszamy załadowanie na CPU
model.load_state_dict(torch.load("combined_fingerprint_model_with_minutiae.pth", map_location=torch.device('cpu')))
model.eval()


# Funkcje pomocnicze
def hash_password(password):
    """Hashuje hasło przy użyciu SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_password, provided_password):
    """Weryfikuje hasło poprzez porównanie hashy."""
    return stored_password == hash_password(provided_password)

def preprocess_fingerprint(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.medianBlur(binary, 5)
    size = np.size(binary)
    skel = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    while not done:
        eroded = cv2.erode(binary, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary, temp)
        skel = cv2.bitwise_or(skel, temp)
        binary = eroded.copy()
        zeros = size - cv2.countNonZero(binary)
        if zeros == size:
            done = True
    return skel

def extract_minutiae(skeleton):
    minutiae = []
    height, width = skeleton.shape
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if skeleton[y, x] == 255:
                neighborhood = [
                    skeleton[y - 1, x - 1], skeleton[y - 1, x], skeleton[y - 1, x + 1],
                    skeleton[y, x - 1],                             skeleton[y, x + 1],
                    skeleton[y + 1, x - 1], skeleton[y + 1, x], skeleton[y + 1, x + 1]
                ]
                num_neighbors = sum(1 for pixel in neighborhood if pixel == 255)
                if num_neighbors == 1:
                    minutiae.append("ending")
                elif num_neighbors == 3:
                    minutiae.append("bifurcation")
    return len([m for m in minutiae if m == "ending"]), len([m for m in minutiae if m == "bifurcation"])

def extract_geometric_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return [0, 0, 0, 0]
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / h
    extent = area / (w * h)
    return [area, perimeter, aspect_ratio, extent]

# Flask app
app = Flask(__name__, template_folder="templates")
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
        password = request.form["password"]
        hashed_password = hash_password(password)
        file = request.files["fingerprint"]
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        skeleton = preprocess_fingerprint(file_path)
        geometry_features = extract_geometric_features(file_path)
        minutiae_features = extract_minutiae(skeleton)

        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 128)) / 255.0
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            cnn_features = model.cnn(image_tensor).flatten().numpy()

        np.savetxt(os.path.join(DATABASE_FOLDER, f"{username}_cnn.txt"), cnn_features, fmt="%.5f")
        np.savetxt(os.path.join(DATABASE_FOLDER, f"{username}_geometry.txt"), geometry_features, fmt="%.5f")
        np.savetxt(os.path.join(DATABASE_FOLDER, f"{username}_minutiae.txt"), minutiae_features, fmt="%.5f")
        with open(os.path.join(DATABASE_FOLDER, f"{username}_password.txt"), "w") as f:
            f.write(hashed_password)

        return render_template("register_success.html", username=username, geometry_features=geometry_features, minutiae_features=minutiae_features)

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        file = request.files["fingerprint"]
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Ścieżki do danych użytkownika
        password_file = os.path.join(DATABASE_FOLDER, f"{username}_password.txt")
        cnn_file = os.path.join(DATABASE_FOLDER, f"{username}_cnn.txt")
        geometry_file = os.path.join(DATABASE_FOLDER, f"{username}_geometry.txt")
        minutiae_file = os.path.join(DATABASE_FOLDER, f"{username}_minutiae.txt")

        # Sprawdzenie, czy użytkownik istnieje
        if not (os.path.exists(password_file) and os.path.exists(cnn_file) and os.path.exists(geometry_file) and os.path.exists(minutiae_file)):
            return render_template("login_failure.html", message="Użytkownik nie istnieje.", show_distances=False)

        # Weryfikacja hasła
        with open(password_file, "r") as f:
            stored_password = f.read().strip()
        if not verify_password(stored_password, password):
            return render_template("login_failure.html", message="Niepoprawne hasło.", show_distances=False)

        # Wczytanie zapisanych cech
        registered_cnn_features = np.loadtxt(cnn_file)
        registered_geometry_features = np.loadtxt(geometry_file)
        registered_minutiae_features = np.loadtxt(minutiae_file)

        # Ekstrakcja nowych cech
        skeleton = preprocess_fingerprint(file_path)
        new_geometry_features = extract_geometric_features(file_path)
        new_minutiae_features = extract_minutiae(skeleton)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 128)) / 255.0
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            new_cnn_features = model.cnn(image_tensor).flatten().numpy()

        # Obliczanie odległości
        cnn_distance = np.linalg.norm(registered_cnn_features - new_cnn_features)
        geometry_distance = np.linalg.norm(registered_geometry_features - new_geometry_features)
        minutiae_distance = np.linalg.norm(registered_minutiae_features - np.array(new_minutiae_features))

        # Sprawdzenie progów
        if cnn_distance < 10.0 and geometry_distance < 5.0 and minutiae_distance < 5.0:
            session["username"] = username
            return render_template("login_success.html", username=username, cnn_distance=cnn_distance, geometry_distance=geometry_distance, minutiae_distance=minutiae_distance)
        else:
            return render_template(
                "login_failure.html", 
                message="Nieprawidłowe dane biometryczne.", 
                show_distances=True,
                cnn_distance=cnn_distance, 
                geometry_distance=geometry_distance, 
                minutiae_distance=minutiae_distance
            )

    return render_template("login.html")



@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
