# 🧠 Facial Emotion & Liveness Recognition

** 🎭 Facial Emotion & Liveness Recognition** is a high-performance Convolutional Neural Network (CNN) application for real-time FER, optimized for low-latency web environments. Built on PyTorch using a fine-tuned MobileNetV2 architecture and Transfer Learning. Key achievements include implementing geometric Anti-Spoofing (Liveness Detection) and optimizing the model for 93.45% accuracy via specialized data handling techniques.  

👉 **Live Demo on Hugging Face:**  
https://huggingface.co/spaces/Giumi10/facial-emotion-recognition

---

## ⚙️ Features

- 🎥 **Real-time webcam processing** (browser or local)
- 😄 **Emotion Recognition** (7 classes):
- angry, disgust, fear, happy, neutral, sad, surprise
- 👁️ **Blink-based liveness detection**
- 😀 **Emoji overlay** for each predicted emotion
- 🔋 **Lightweight MobileNetV2 model**

---

## 🧰 Technologies Used

- **Python 3**
- **PyTorch** – CNN model training + inference
- **MobileNetV2** – Feature extractor
- **MediaPipe Face Mesh** – Blink + eye landmark detection
- **OpenCV** – Frame processing & emoji overlay
- **Gradio** – Live camera UI
- **Hugging Face Spaces** – Hosting environment

---

## 🧠 Training the Model (Optional)

The emotion classification model was trained using the **FER+ (Facial Expression Recognition Plus)** dataset, a widely-used dataset for facial emotion recognition tasks.

📁 The training script is available in this repo as: train_model.py

---

## 📂 Dataset Info

- Dataset used: **FER+** (you can find it [here](https://www.kaggle.com/datasets/msambare/fer2013))
- Images are grayscale, 48x48 pixels, with labeled facial emotions.
- The dataset is split into `train` and `test`.

> ⚠️ **Note:** The dataset is not included in this repository due to high storage memory. Please download it manually if you want to retrain the model.

## 🚀 How to Train

To retrain the model locally:

```bash
python train_model.py
```
🧪 After training, the model will be saved as: models/emotion_model_best.pt

## 🚀 Try It Online

▶️ Test in your browser, no installation needed:

👉 https://huggingface.co/spaces/Giumi10/facial-emotion-recognition

Just allow camera access and the app will start automatically.

⚠️ Performance Note
This demo runs on a Free Cloud CPU. Due to hardware limitations, the processing speed is optimized for stability rather than high frame rates. You may experience slight latency.

---

## 🖥️ Run Locally (Recommended)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/StefanGiumanca/Facial-Emotion-Recognition.git
cd Facial-Emotion-Recognition
```
### 2️⃣ (Optional) Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # For macOS / Linux
# venv\Scripts\activate         # For Windows
```
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ RUN the app
```bash
cd src
python3 run_webcam.py
