import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
from collections import deque

# Load the trained model
model = load_model("emotion_model.h5")

# Emotion labels (same order as training)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Emotion to Stress Mapping
emotion_to_stress = {
    'happy': 'Low',
    'neutral': 'Medium',
    'sad': 'High',
    'angry': 'High',
    'fear': 'High',
    'disgust': 'High',
    'surprise': 'Medium'
}

# Suggestions
stress_suggestions = {
    'Low': "Keep up the good mood! 😊",
    'Medium': "Take a short break or breathe deeply 🧘‍♂️",
    'High': "Try relaxation: deep breaths, stretch, or a walk 🌿",
    'Unknown': "Stay mindful and calm 🕊️"
}

# Initialize face detector
detector = MTCNN()
cap = cv2.VideoCapture(0)

emotion_buffer = deque(maxlen=10)
confidence_buffer = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector.detect_faces(frame)
    if faces:
        x, y, w, h = faces[0]['box']
        x, y = max(0, x), max(0, y)
        face_roi = frame[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.resize(face_gray, (48, 48))
        face_gray = face_gray / 255.0
        face_gray = np.expand_dims(face_gray, axis=(0, -1))

        preds = model.predict(face_gray)[0]
        dom_emotion = emotion_labels[np.argmax(preds)]
        confidence = np.max(preds) * 100

        emotion_buffer.append(dom_emotion)
        confidence_buffer.append(confidence)

        emotion = max(set(emotion_buffer), key=emotion_buffer.count)
        stress = emotion_to_stress.get(emotion, "Unknown")
        suggestion = stress_suggestions.get(stress, "Stay calm 🕊️")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, f"Emotion: {emotion} ({confidence:.1f}%)", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Stress: {stress}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Tip: {suggestion}", (50, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    else:
        cv2.putText(frame, "No face detected!", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Real-Time Emotion & Stress Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

