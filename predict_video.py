# ================= CLEAN TERMINAL =================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger('absl').setLevel(logging.ERROR)

# ================= IMPORTS =================
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

from pose_engine413 import full_body_feedback
from voice413 import speak

# ================= CONFIG =================
WINDOW_SIZE = 30
PREDICTION_THRESHOLD = 0.80

print("Initializing AI Trainer...")
model = load_model("model/zumba_lstm_model.h5")

# ================= MEDIAPIPE =================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=0,
    smooth_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_drawing = mp.solutions.drawing_utils

# ================= KEYPOINT EXTRACTION =================
def extract_keypoints(frame):
    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        return np.array(keypoints), results

    return np.zeros(33 * 3), results

# ================= VIDEO MODE =================
def predict_saved_video(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []

    print("\nProcessing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, _ = extract_keypoints(frame)
        sequence.append(keypoints)

    cap.release()

    if len(sequence) < WINDOW_SIZE:
        print("Video too short.")
        return

    X = []
    for i in range(len(sequence) - WINDOW_SIZE):
        X.append(sequence[i:i+WINDOW_SIZE])

    X = np.array(X)

    predictions = model.predict(X, verbose=0)
    avg_pred = float(np.mean(predictions))

    print("\n===== RESULT =====")
    print(f"Score: {avg_pred:.3f}")
    print("CORRECT" if avg_pred > PREDICTION_THRESHOLD else "INCORRECT")

# ================= WEBCAM MODE =================
def webcam_mode():
    cap = cv2.VideoCapture(0)

    sequence = []
    pred_buffer = []
    feedback_buffer = []

    reps = 0
    state = "up"

    last_voice = ""
    last_time = 0

    frame_skip = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        keypoints, results = extract_keypoints(frame)

        if not results.pose_landmarks:
            cv2.putText(frame, "No pose detected",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
            cv2.imshow("Zumba AI Trainer", frame)
            continue

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        lm = results.pose_landmarks.landmark

        # ===== SEQUENCE =====
        sequence.append(keypoints)
        if len(sequence) > WINDOW_SIZE:
            sequence.pop(0)

        if len(sequence) < WINDOW_SIZE:
            cv2.putText(frame, "Initializing...",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 0), 2)
            cv2.imshow("Zumba AI Trainer", frame)
            continue

        # ===== PERFORMANCE =====
        frame_skip += 1
        if frame_skip % 2 != 0:
            cv2.imshow("Zumba AI Trainer", frame)
            continue

        # ===== MODEL =====
        pred = model.predict(np.expand_dims(sequence, 0), verbose=0)[0][0]
        pred_buffer.append(pred)

        if len(pred_buffer) > 10:
            pred_buffer.pop(0)

        avg_pred = np.mean(pred_buffer)

        # ===== FEEDBACK =====
        feedback, errors = full_body_feedback(lm)

        feedback_buffer.append(feedback)
        if len(feedback_buffer) > 10:
            feedback_buffer.pop(0)

        stable_feedback = max(set(feedback_buffer), key=feedback_buffer.count)

        # ===== SCORE =====
        score = int((1 - errors / 5) * 100)
        score = max(0, score)

        # ===== REPS =====
        if errors > 2 and state == "up":
            state = "down"
        elif errors <= 1 and state == "down":
            reps += 1
            state = "up"

        # ===== VOICE =====
        if stable_feedback != last_voice and time.time() - last_time > 3:
            speak(stable_feedback)
            last_voice = stable_feedback
            last_time = time.time()

        # ===== DISPLAY =====
        status = "CORRECT" if avg_pred > PREDICTION_THRESHOLD else "INCORRECT"
        color = (0, 255, 0) if score > 80 else (0, 0, 255)

        cv2.putText(frame, f"{status} ({avg_pred:.2f})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2)

        cv2.putText(frame, f"Score: {score}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

        cv2.putText(frame, stable_feedback,
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)

        cv2.putText(frame, f"Reps: {reps}",
                    (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 0), 2)

        cv2.imshow("Zumba AI Trainer PRO", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ================= MAIN =================
if __name__ == "__main__":

    print("\n====== ZUMBA AI TRAINER SYSTEM ======")
    print("1. Webcam Mode")
    print("2. Video Mode")

    choice = input("Enter your choice: ")

    if choice == "1":
        webcam_mode()

    elif choice == "2":
        video_path = input("Enter video path: ")
        predict_saved_video(video_path)

    else:
        print("Invalid choice")