from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from pymongo import MongoClient
import numpy as np
import os
from datetime import datetime

# ------------------------------------------------------
# 🔧 Configuration
# ------------------------------------------------------
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'model.h5'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------------------------------------
# 🧠 Train CNN Model (only if not exists)
# ------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    print("Training model... please wait (first run only)...")

    # Create small CNN
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')  # Example: 3 classes
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.save(MODEL_PATH)

model = load_model(MODEL_PATH)

# ------------------------------------------------------
# 🗃️ MongoDB Setup
# ------------------------------------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["image_classification_db"]
collection = db["predictions"]

# Example labels — change according to your dataset
labels = ['cat', 'dog', 'car']

# ------------------------------------------------------
# 🖥️ Simple Frontend HTML (embedded)
# ------------------------------------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>AI Image Classifier</title>
<style>
body {
  font-family: Arial, sans-serif;
  background: #f4f4f9;
  text-align: center;
  margin-top: 80px;
}
.container {
  background: white;
  width: 420px;
  margin: auto;
  padding: 30px;
  border-radius: 15px;
  box-shadow: 0 0 10px #ccc;
}
button {
  background-color: #007bff;
  color: white;
  padding: 10px 15px;
  border: none;
  border-radius: 10px;
  cursor: pointer;
}
button:hover {
  background-color: #0056b3;
}
img {
  margin-top: 15px;
  width: 250px;
  border-radius: 10px;
}
.result {
  margin-top: 20px;
  background: #e9ecef;
  padding: 15px;
  border-radius: 10px;
}
</style>
</head>
<body>
<div class="container">
  <h2>🧠 AI Image Classification</h2>
  <form id="uploadForm" enctype="multipart/form-data">
    <input type="file" name="file" accept="image/*" required><br><br>
    <button type="submit">Predict</button>
  </form>
  <div id="result"></div>
</div>

<script>
document.getElementById('uploadForm').onsubmit = async (e) => {
  e.preventDefault();
  const fileInput = e.target.querySelector('input[type=file]');
  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  document.getElementById('result').innerHTML = "<p>Processing...</p>";

  const response = await fetch('/predict', { method: 'POST', body: formData });
  const data = await response.json();
  
  if(data.error) {
    document.getElementById('result').innerHTML = `<p style='color:red;'>${data.error}</p>`;
  } else {
    document.getElementById('result').innerHTML = `
      <div class="result">
        <p><strong>Prediction:</strong> ${data.prediction}</p>
        <p><strong>Confidence:</strong> ${data.confidence}%</p>
        <img src="/static/uploads/${data.filename}" alt="uploaded">
      </div>`;
  }
};
</script>
</body>
</html>
"""

# ------------------------------------------------------
# 🌐 Routes
# ------------------------------------------------------
@app.route('/')
def home():
    return render_template_string(HTML_PAGE)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty file name'})

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess the image
    img = load_img(filepath, target_size=(64, 64))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions)
    pred_label = labels[pred_index]
    confidence = float(np.max(predictions)) * 100

    # Save result in MongoDB
    record = {
        "filename": file.filename,
        "prediction": pred_label,
        "confidence": round(confidence, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    collection.insert_one(record)

    return jsonify(record)

# ------------------------------------------------------
# 🚀 Run
# ------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
