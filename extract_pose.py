import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ✅ THIS LINE WAS BROKEN BEFORE — NOW FIXED
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            # if pose not detected → fill zeros
            keypoints = [0] * (33 * 3)

        sequence.append(keypoints)

    cap.release()
    return np.array(sequence)




