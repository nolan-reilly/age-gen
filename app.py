from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import torch.nn.functional as F

app = Flask(__name__)
CORS(app)

class AgeEstimationCNN(torch.nn.Module):
    def __init__(self):
        super(AgeEstimationCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(256)
        self.conv5 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(512)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        fc_input_size = 512 * (256 // 32) * (256 // 32)  # 512 * 8 * 8
        self.fc1 = torch.nn.Linear(fc_input_size, 1024)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(1024, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class GenderEstimationCNN(torch.nn.Module):
    def __init__(self):
        super(GenderEstimationCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(256)
        self.conv5 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(512)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        fc_input_size = 512 * (256 // 32) * (256 // 32)  # same calculation
        self.fc1 = torch.nn.Linear(fc_input_size, 1024)
        self.dropout = torch.nn.Dropout(p=0.5)
        # Single output for binary classification (0=female, 1=male)
        self.fc2 = torch.nn.Linear(1024, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Load age estimation model
age_model = AgeEstimationCNN()
age_model.load_state_dict(torch.load('age_estimation_model.pth', map_location='cpu'))
age_model.eval()

# Load gender classification model
gender_model = GenderEstimationCNN()
gender_model.load_state_dict(torch.load('gender_classification_model.pth', map_location='cpu'))
gender_model.eval()

input_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def crop_to_face(image):
    # Convert PIL Image to numpy array
    image_np = np.array(image)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Load OpenCV's default face detector (Haar cascades)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If faces are found, crop to the first face
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = image_np[y:y + h, x:x + w]
        return Image.fromarray(face)  # Convert back to PIL Image

    # If no face is detected, return the original image
    return image

def generate_age_message(age):
    if 0 <= age <= 2:
        return "Fresh out the womb"
    elif 3 <= age <= 12:
        return "Gremlin"
    elif 13 <= age <= 17:
        return "JIT"
    elif 18 <= age <= 24:
        return "Youngblood"
    elif 25 <= age <= 30:
        return "Developing Unc"
    elif 31 <= age <= 35:
        return "Peak Unc"
    elif 36 <= age <= 50:
        return "NPC"
    elif 51 <= age <= 65:
        return "Old Head"
    elif age < 65:
        return "OG"
    else:
        return "Invalid age provided."

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Open the image and crop it to the face
    image = Image.open(file).convert('RGB')
    face_image = crop_to_face(image)

    # Transform the cropped image
    input_tensor = input_transforms(face_image).unsqueeze(0)

    # Age prediction
    with torch.no_grad():
        age_output = age_model(input_tensor)
    predicted_age = age_output.item()

    # Gender prediction
    with torch.no_grad():
        gender_output = gender_model(input_tensor)
    # Convert logits to probability
    gender_prob = torch.sigmoid(gender_output).item()
    predicted_gender = "Female" if gender_prob > 0.5 else "Male"

    # Generate a message based on the predicted age
    message = generate_age_message(predicted_age)

    return jsonify({
        "predicted_age": predicted_age,
        "message": message,
        "predicted_gender": predicted_gender
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
