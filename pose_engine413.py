import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])

    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle

    return angle


def normalize_landmarks(lm):
    hip_x = (lm[23].x + lm[24].x) / 2
    hip_y = (lm[23].y + lm[24].y) / 2

    normalized = []
    for l in lm:
        normalized.append([l.x - hip_x, l.y - hip_y])

    return normalized


def full_body_feedback(landmarks):
    lm = normalize_landmarks(landmarks)

    issues = []

    # Arms
    if calculate_angle(lm[11], lm[13], lm[15]) < 150:
        issues.append(("Raise left arm", 2))
    if calculate_angle(lm[12], lm[14], lm[16]) < 150:
        issues.append(("Raise right arm", 2))

    # Legs
    if calculate_angle(lm[23], lm[25], lm[27]) < 160:
        issues.append(("Straighten left leg", 1))
    if calculate_angle(lm[24], lm[26], lm[28]) < 160:
        issues.append(("Straighten right leg", 1))

    # Back posture
    shoulder_mid = [(lm[11][0]+lm[12][0])/2, (lm[11][1]+lm[12][1])/2]
    hip_mid = [(lm[23][0]+lm[24][0])/2, (lm[23][1]+lm[24][1])/2]

    if abs(shoulder_mid[0] - hip_mid[0]) > 0.05:
        issues.append(("Keep your back straight", 3))

    issues.sort(key=lambda x: -x[1])

    feedback = issues[0][0] if issues else "Perfect posture"
    errors = len(issues)

    return feedback, errors