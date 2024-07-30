import cv2
import mediapipe as mp
import numpy as np
from utils1 import calculate_angle, put_text_rectangle

# Initialize the drawing utilities for MediaPipe
mp_drawing: mp.solutions.drawing_utils = mp.solutions.drawing_utils
mp_pose: mp.solutions.pose = mp.solutions.pose

# Initialize the webcam capture
webcam: cv2.VideoCapture = cv2.VideoCapture(0)  

# Create a Pose instance with specified minimum detection and tracking confidence
with mp_pose.Pose(min_detection_confidence = 0.7, min_tracking_confidence = 0.7) as pose:
    while webcam.isOpened():
        # Read a frame from the webcam
        response: bool
        frame: np.ndarray
        response, frame = webcam.read()

        # Check if the frame was read successfully
        if not response:
            print("Failed to read from webcam. Exiting...")
            break

        # Convert the frame from BGR to RGB format for MediaPipe processing
        frame_rgb: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make the frame non-writable to improve performance
        frame_rgb.flags.writeable = False

        # Process the frame to detect pose landmarks
        results: mp.solutions.pose.Pose = pose.process(frame_rgb)

        # Make the frame writable again
        frame_rgb.flags.writeable = True
        # Convert the frame back to BGR format for OpenCV processing
        frame_bgr: np.ndarray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Attempt to extract landmarks from the pose detection results
        try:
            landmarks: mp.solutions.pose.PoseLandmark = results.pose_landmarks.landmark

            # Extract the coordinates of the shoulder, elbow, and wrist landmarks
            shoulder: list = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow: list = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist: list = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate the angle between the shoulder, elbow, and wrist
            angle: float = calculate_angle(a=shoulder, b=elbow, c=wrist)

            # Determine the posture based on the angle
            if angle < 160:
                posture: str = "Bad Posture"
                color: tuple = (0, 0, 255)  # Red color for bad posture
            else:
                posture: str = "Good Posture"
                color: tuple = (0, 255, 0)  # Green color for good posture

            # Display the posture status on the frame
            frame_bgr, _ = put_text_rectangle(frame_bgr, posture, (50, 50), colour_rectangle=color, border=2)

        except AttributeError:
            pass

        # Draw the skeletal outline on the frame
        mp_drawing.draw_landmarks(
             frame_bgr,
             results.pose_landmarks,
             mp_pose.POSE_CONNECTIONS,
             mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
             mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
         )


        # Display the frame
        cv2.imshow('Webcam', frame_bgr)

        # Check for the 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    webcam.release()
    cv2.destroyAllWindows()
