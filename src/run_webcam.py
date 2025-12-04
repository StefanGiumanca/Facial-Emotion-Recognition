import cv2
import torch 
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import mediapipe as mp
import time

# Liveness Detection 

mp_face_detection = mp.solutions.face_detection # Bounding box
mp_face_mesh = mp.solutions.face_mesh # landmarks
face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode = False,
        max_num_faces = 1,
        refine_landmarks = True,
        min_detection_confidence = 0.5
    )

LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [263, 387, 385, 362, 380, 373]


blink_threshold = 0.21
spoof_timeout = 10.0 # Maximum time without blink/moving EAR - Eye Aspect Ratio

def euclidean(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(eye_points):
        A = euclidean(eye_points[1], eye_points[5])
        B = euclidean(eye_points[2], eye_points[4])
        C = euclidean(eye_points[0], eye_points[3])
        ear = (A + B) / (2.0 * C)
        return ear

# Checking the CPU/GPU the user has
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: Apple MPS GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: NVIDIA CUDA ({torch.cuda.get_device_name(0)})")
else:
    device = torch.device("cpu")
    print('Using device: CPU')

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

emoji_path = "../data/emojis"
emoji_images = {}

for emotion in classes:
    path = f"{emoji_path}/{emotion}.png"
    emoji = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if emoji is not None:
        emoji_images[emotion] = emoji
    else:
        print(f"[WARNING] Could not load emoji for {emotion}")


model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
model.load_state_dict(torch.load("../models/emotion_model_best.pt", map_location=device))
model.to(device)
model.eval() # !!! Evaluate the model

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Camera not found.')
    exit()

print("Webcam ON - press 'q' to quit.")

last_blink_time = time.time()
is_closed = False   # Eyes

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results_mesh = face_mesh.process(frame_rgb)
    results_detection = face_detection.process(frame_rgb)

    is_live = False

    if results_mesh.multi_face_landmarks:
            landmarks = results_mesh.multi_face_landmarks[0]
            ih, iw, _ = frame.shape

            left_eye = []
            for idx in LEFT_EYE_LANDMARKS:
                x = int(landmarks.landmark[idx].x * iw)
                y = int(landmarks.landmark[idx].y * ih)
                left_eye.append((x,y))

            right_eye = []
            for idx in RIGHT_EYE_LANDMARKS:
                x = int(landmarks.landmark[idx].x * iw)
                y = int(landmarks.landmark[idx].y * ih)
                right_eye.append((x,y))

            #Calculate the EAR (Eye Aspect Ratio)
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < blink_threshold and not is_closed:
                is_closed = True
            elif avg_ear >= blink_threshold and is_closed: # the user blinked, so we reset the timer
                last_blink_time = time.time()
                is_closed = False
            
    if time.time() - last_blink_time < spoof_timeout:
        is_live = True
        liveness_status = "Status: Live"
        liveness_color = (0, 128, 0) # Green / The person is live
    else:
        is_live = False
        liveness_status = f"Status = SPOOFING DETECTED *Blink* ({int(time.time() - last_blink_time)}s)"
        liveness_color = (0, 0, 255)

    font_size = 1.3
    font_thickness = 2
        
    (text_w, text_h), baseline = cv2.getTextSize(liveness_status, cv2.FONT_HERSHEY_DUPLEX, font_size, font_thickness)
        
    text_anchor_x = 10 
    text_anchor_y = 45 

    bg_start_x = text_anchor_x - 5
    bg_start_y = text_anchor_y - text_h - 5
        
    bg_end_x = text_anchor_x + text_w + 5
    bg_end_y = text_anchor_y + baseline + 5 
        
    cv2.rectangle(frame, (bg_start_x, bg_start_y), (bg_end_x, bg_end_y), liveness_color, -1) 
    cv2.putText(frame, liveness_status, (text_anchor_x, text_anchor_y), 
    cv2.FONT_HERSHEY_DUPLEX, font_size, (255, 255, 255), font_thickness)

    if results_mesh.multi_face_landmarks:
        eye_color = (0, 255, 0) # Verde (Live)
        if not is_live:
             eye_color = (0, 0, 255) # Roșu (Spoofing)
        for (x_eye, y_eye) in left_eye + right_eye:
            cv2.circle(frame, (x_eye, y_eye), 1, eye_color, -1) # -1 pentru cerc plin

    if results_detection.detections and is_live: 
        for detection in results_detection.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape

            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            face = frame[y:y+h, x:x+w]
            if face.size == 0: 
                continue

            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            input_tensor = transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
                emotion = classes[pred.item()]
                confidence = torch.softmax(output, dim=1)[0][pred.item()].item()
                label = f"{emotion} ({confidence*100:.1f}%)"

                emoji = emoji_images.get(emotion)
                emoji_size = h // 2
                
                if emoji is not None and emoji_size > 20: 
                    emoji_resized = cv2.resize(emoji, (emoji_size, emoji_size), interpolation=cv2.INTER_AREA)

                    x_start = x 
                    y_start = y - emoji_size - 10 

                    y1 = max(0, y_start)
                    y2 = min(frame.shape[0], y_start + emoji_size)
                    x1 = max(0, x_start)
                    x2 = min(frame.shape[1], x_start + emoji_size)
                    
                    h_actual = y2 - y1
                    w_actual = x2 - x1
                    
                    if h_actual > 0 and w_actual > 0:
                        emoji_crop = emoji_resized[0:h_actual, 0:w_actual] 
                        

                        if emoji_crop.shape[2] == 4: 
                            alpha_emoji = emoji_crop[:, :, 3] / 255.0
                            alpha_frame = 1.0 - alpha_emoji
                            
                            for c in range(3): 
                                frame[y1:y2, x1:x2, c] = (
                                    alpha_emoji * emoji_crop[:, :, c] +
                                    alpha_frame * frame[y1:y2, x1:x2, c]
                                )
                
                print("Tensor shape:", input_tensor.shape)
                print("Pred:", emotion, " | Conf:", round(confidence*100, 2), "%")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.imshow("Facial Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()