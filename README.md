# 🧠 Facial Emotion & Liveness Recognition

**Facial Emotion & Liveness Recognition** is a real-time AI application that detects facial emotions and verifies if the user is *live* (not a spoof/photo) using blink detection.  

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
3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
4️⃣ RUN the app
```bash
cd src
python3 run_webcam.py
