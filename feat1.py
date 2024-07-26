import cv2
import mediapipe as mp
import numpy as np
from utils1 import calculate_angle, put_text_rectangle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

webcam = cv2.VideoCapture(0)  

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    while webcam.isOpened():
        response, frame = webcam.read()

        if not response:
            print("Failed to read from webcam. Exiting...")
            break

        # Recolor Image from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        # Make detection
        results = pose.process(frame_rgb)

        # Recolor back to BGR format
        frame_rgb.flags.writeable = True
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            angle = calculate_angle(a=shoulder, b=elbow, c=wrist)

            # Determine posture
            if angle < 160:
                posture = "Bad Posture"
                color = (0, 0, 255)  # Red for bad posture
            else:
                posture = "Good Posture"
                color = (0, 255, 0)  # Green for good posture

            # Display posture status on the frame
            frame_bgr, _ = put_text_rectangle(frame_bgr, posture, (50, 50), colour_rectangle=color, border=2)

        except AttributeError:
            pass

        #This line is for the skeletal outline
        mp_drawing.draw_landmarks(
             frame_bgr,
             results.pose_landmarks,
             mp_pose.POSE_CONNECTIONS,
             mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
             mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
         )


        cv2.imshow('Webcam', frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()

