# AI-Image-Classification-Web-App
A full-stack AI-powered web application for image classification using Deep Learning (CNN).
Built with Flask, TensorFlow, and MongoDB, it allows users to upload an image and get instant classification results along with prediction confidence.

**Project Demo**

🚀 Live Demo (Localhost):
http://127.0.0.1:5000/

When you upload an image, it:

Runs through a trained Convolutional Neural Network (CNN)

Predicts the most likely class (e.g., Cat, Dog, Car)

Displays prediction + confidence

Saves results to MongoDB for analytics/logging

**Features**

✅ Upload any image (JPG, PNG, etc.)
✅ Classify using a CNN model (TensorFlow/Keras)
✅ Real-time results with confidence score
✅ Stores predictions in MongoDB
✅ Simple & clean frontend (HTML + JS + CSS inside Flask)
✅ Single-file deployment (app.py)

**Tech Stack**
| Component         | Technology                                  |
| ----------------- | ------------------------------------------- |
| **Frontend**      | HTML5, CSS3, Vanilla JS                     |
| **Backend**       | Flask (Python)                              |
| **Deep Learning** | TensorFlow / Keras                          |
| **Database**      | MongoDB                                     |
| **Libraries**     | numpy, pillow, flask-cors                   |
| **Model**         | Custom CNN trained for image classification |


**Folder Structure**
ImageClassificationApp/
│
├── app.py                  # Main full-stack Flask application
├── static/
│   └── uploads/            # Stores uploaded images
├── model.h5                # Saved CNN model
└── README.md               # Documentation (this file)


**Installation & Setup**
1️⃣ Clone the Repository
git clone https://github.com/yourusername/Image-Classification-NN.git
cd Image-Classification-NN

2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

3️⃣ Install Dependencies
pip install flask flask-cors tensorflow pymongo pillow numpy

4️⃣ Start MongoDB

Make sure MongoDB service is running locally:

mongod

5️⃣ Run the Application
python app.py

6️⃣ Open the App

Visit 👉 http://127.0.0.1:5000 in your web browser.

Model Information

A simple CNN with:

2 Convolutional + MaxPooling layers

1 Dense hidden layer

Softmax output for classification

Input image size: 64×64×3

Optimizer: Adam

Loss: Categorical Crossentropy

Accuracy: ~90%+ (depending on dataset)

You can retrain the model by replacing the CNN training section in app.py with your dataset.

**Example Output**
🖼️ Uploaded Image:
dog.jpg

🧠 Model Output:
{
  "filename": "dog.jpg",
  "prediction": "dog",
  "confidence": 97.83,
  "timestamp": "2025-10-08 10:23:55"
}

🖥️ Web Interface:
Prediction: Dog
Confidence: 97.83%


And your uploaded image is displayed below.

MongoDB Record Example
{
  "_id": { "$oid": "6521e95f3a5a2a47dc7a8c2b" },
  "filename": "cat_1.jpg",
  "prediction": "cat",
  "confidence": 95.42,
  "timestamp": "2025-10-08 09:47:11"
}

**How It Works**

Upload → User selects an image.

Preprocess → Image resized to 64×64 & normalized.

Predict → CNN model predicts the class label.

Store → Result saved to MongoDB with timestamp.

Display → Frontend shows prediction + confidence + image.

**Future Enhancements**

 Add MySQL support (optional backend database)

 Integrate pre-trained models (ResNet, MobileNet)

 User authentication for tracking history

 Deploy on AWS / Render / Heroku

 Add frontend with React or Streamlit

 
