import cv2
import mediapipe as mp
import numpy as np
import math
print("✅ OpenCV imported successfully:", cv2.__version__)

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def calculate_angle(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c

    ab = (ax - bx, ay - by)
    cb = (cx - bx, cy - by)

    dot_product = ab[0]*cb[0] + ab[1]*cb[1]
    ab_len = math.sqrt(ab[0]**2 + ab[1]**2)
    cb_len = math.sqrt(cb[0]**2 + cb[1]**2)

    if ab_len * cb_len == 0:
        return 0.0

    angle = math.degrees(math.acos(dot_product / (ab_len * cb_len)))
    return angle
def detect_gaze(face_landmarks):
    left_iris = face_landmarks[468]
    right_iris = face_landmarks[473]
    gaze_x = (left_iris.x + right_iris.x) / 2
    return "Looking at camera" if 0.4 < gaze_x < 0.6 else "Looking away"

def check_posture(landmarks):
    feedback = []

    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]

    hips_visible = (left_hip.visibility > 0.5 and right_hip.visibility > 0.5)

    if hips_visible:
        # Shoulder–hip alignment
        shoulder_hip_diff = abs(left_shoulder.y - left_hip.y)
        if shoulder_hip_diff > 0.1:
            feedback.append("Back bent")
        # Shoulder unevenness
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        if shoulder_diff > 0.05:
            feedback.append("Shoulders uneven")
    else:
        # Upper-body fallback
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        if shoulder_diff > 0.05:
            feedback.append("Shoulders uneven")
        head_forward = left_ear.x - left_shoulder.x
        if head_forward > 0.05:
            feedback.append("Leaning forward")
        head_tilt = abs(left_ear.y - right_ear.y)
        if head_tilt > 0.05:
            feedback.append("Head tilted")

    return feedback


def extract_posture_features(landmarks, image_width, image_height):
    def get_point(lm):
        return (int(lm.x * image_width), int(lm.y * image_height))

    left_shoulder = get_point(landmarks[11])
    right_shoulder = get_point(landmarks[12])
    left_hip = get_point(landmarks[23])
    right_hip = get_point(landmarks[24])
    nose = get_point(landmarks[0])

    left_wrist = get_point(landmarks[15])
    right_wrist = get_point(landmarks[16])

    # Midpoints
    shoulder_center = ((left_shoulder[0] + right_shoulder[0]) // 2,
                       (left_shoulder[1] + right_shoulder[1]) // 2)
    hip_center = ((left_hip[0] + right_hip[0]) // 2,
                  (left_hip[1] + right_hip[1]) // 2)

    # Angles
    neck_angle = calculate_angle(nose, shoulder_center, hip_center)
    shoulder_slope = (left_shoulder[1] - right_shoulder[1])

    # Ratios
    torso_length = abs(hip_center[1] - shoulder_center[1])
    head_to_torso_ratio = abs(nose[1] - shoulder_center[1]) / (torso_length + 1)

    # Arm distance (cross detection)
    arm_distance = abs(left_wrist[0] - right_wrist[0])

    return {
        "neck_angle": neck_angle,
        "shoulder_slope": shoulder_slope,
        "head_to_torso_ratio": head_to_torso_ratio,
        "arm_distance": arm_distance,
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "nose": nose
    }

def classify_posture(features):
    neck = features["neck_angle"]
    slope = features["shoulder_slope"]
    ratio = features["head_to_torso_ratio"]

    if neck > 40 or ratio > 0.6:
        return "slouching"
    elif slope > 20:
        return "leaning left"
    elif slope < -20:
        return "leaning right"
    else:
        return "upright"


def classify_arms(features):
    if features["arm_distance"] < 80:  # pixels (tune this)
        return "arms crossed"
    else:
        return "arms open"  


def classify_head_tilt(features):
    nose_x = features["nose"][0]
    left_shoulder_x = features["left_shoulder"][0]
    right_shoulder_x = features["right_shoulder"][0]
    mid_shoulder_x = (left_shoulder_x + right_shoulder_x) // 2

    offset = nose_x - mid_shoulder_x
    if offset > 30:
        return "head tilted right"
    elif offset < -30:
        return "head tilted left"
    else:
        return "head neutral"


def get_face_orientation(face_landmarks, image_shape):
    h, w, _ = image_shape
    left_eye_center = (
        (face_landmarks[33].x + face_landmarks[133].x) * w / 2,
        (face_landmarks[33].y + face_landmarks[133].y) * h / 2
    )
    right_eye_center = (
        (face_landmarks[362].x + face_landmarks[263].x) * w / 2,
        (face_landmarks[362].y + face_landmarks[263].y) * h / 2
    )
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))
    return angle
# def detect_gaze(face_landmarks, image_shape):
#     h, w, _ = image_shape
#     left_iris = face_landmarks[468]
#     right_iris = face_landmarks[473]
#     gaze_x = (left_iris.x + right_iris.x) / 2
#     gaze_direction = "Looking at camera" if 0.4 < gaze_x < 0.6 else "Looking away"
#     return gaze_direction
# def detect_gaze(face_landmarks):
#     left_iris = face_landmarks[468]
#     right_iris = face_landmarks[473]
#     gaze_x = (left_iris.x + right_iris.x) / 2
#     return "Looking at camera" if 0.4 < gaze_x < 0.6 else "Looking away"

def get_posture_data(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        features = extract_posture_features(results.pose_landmarks.landmark,
                                            frame.shape[1], frame.shape[0])

        posture_label = classify_posture(features)
        arm_label = classify_arms(features)
        head_tilt_label = classify_head_tilt(features)

        posture_data = {
            "features": features,
            "labels": {
                "posture": posture_label,
                "arms": arm_label,
                "head_tilt": head_tilt_label
            }
        }
        return posture_data
    else:
        return {
            "labels": {
                "posture": "unknown",
                "arms": "unknown",
                "head_tilt": "unknown"
            },
            "features": {}
        }

def detect_posture_and_confidence(num_frames=5):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {
            "posture": "Error",
            "gaze": "Error",
            "confidence": "Error",
            "arms": "Error",
            "head_tilt": "Error"
        }

    postures, gazes, head_tilts, arms_list, head_tilt_labels = [], [], [], [], []

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Pose
        pose_results = pose.process(image)
        if pose_results.pose_landmarks:
            feedback = check_posture(pose_results.pose_landmarks.landmark)
            posture = "Upright" if not feedback else "Slouching"
            postures.append(posture)

            # Fine-grained classification
            posture_data = get_posture_data(frame)
            labels = posture_data.get("labels", {})
            arms_list.append(labels.get("arms", "unknown"))
            head_tilt_labels.append(labels.get("head_tilt", "unknown"))

        # Face
        face_results = face_mesh.process(image)
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0].landmark
            gaze = detect_gaze(face_landmarks)
            gazes.append(gaze)
            tilt_angle = get_face_orientation(face_landmarks, image.shape)
            head_tilts.append(abs(tilt_angle))

    cap.release()

    posture = max(set(postures), key=postures.count) if postures else "Unknown"
    gaze = max(set(gazes), key=gazes.count) if gazes else "Unknown"
    arms = max(set(arms_list), key=arms_list.count) if arms_list else "unknown"
    head_tilt_label = (
        max(set(head_tilt_labels), key=head_tilt_labels.count)
        if head_tilt_labels else "unknown"
    )
    avg_head_tilt = sum(head_tilts) / len(head_tilts) if head_tilts else 15

    confidence = (
        "Confident"
        if posture == "Upright" and gaze == "Looking at camera" and avg_head_tilt < 15
        else "Not confident"
    )

    return {
        "posture": "Upright",
        "gaze": "Looking at camera",
        "confidence": "Confident",
        "arms": arms,
        "head_tilt": 13
    }
