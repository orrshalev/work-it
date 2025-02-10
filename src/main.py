import cv2
import mediapipe as mp
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_drawing

DETECTION_CONFIDENCE, TRACKING_CONFIDENCE = 0.5, 0.5

pose = mp_pose.Pose(
    min_detection_confidence=DETECTION_CONFIDENCE, min_tracking_confidence=0.5
)
mp.tasks.vision.PoseLandmarker

cap = cv2.VideoCapture(0)
i = 0
while cap.isOpened():
    # read frame
    _, frame = cap.read()
    i += 1
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        frame_timestamp_ms = i

        pose_results = pose.process(frame_rgb)

        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        if pose_results.pose_landmarks:
            right_thumb = pose_results.pose_landmarks.landmark[22]
            if right_thumb.visibility > DETECTION_CONFIDENCE:
                print(right_thumb)

        cv2.imshow("Output", frame)
    except Exception as e:
        print(e)
        break

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
