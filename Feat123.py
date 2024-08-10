import cv2
import mediapipe as mp
import math as m
import torch
import torch.nn as nn
import tkinter as tk
import pygame
import threading
import os
class PostureAnalysisModel(nn.Module):
    def __init__(self):
        super(PostureAnalysisModel, self).__init__()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.lmPose = self.mp_pose.PoseLandmark
        self.mp_drawing = mp.solutions.drawing_utils  # Initialize the drawing utilities

    def findDistance(self, x1, y1, x2, y2):
        dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist

    def findAngle(self, x1, y1, x2, y2):
        theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
        degree = (180 / m.pi) * theta
        return degree

    def forward(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = self.pose.process(image_rgb)
        lm = keypoints.pose_landmarks

        if lm is None:
            return None

        # Draw the skeletal outline on the frame
        self.mp_drawing.draw_landmarks(
            image,
            lm,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        # Acquire the landmark coordinates.
        # Left shoulder.
        l_shldr_x = int(lm.landmark[self.lmPose.LEFT_SHOULDER].x * image.shape[1])
        l_shldr_y = int(lm.landmark[self.lmPose.LEFT_SHOULDER].y * image.shape[0])

        # Right shoulder
        r_shldr_x = int(lm.landmark[self.lmPose.RIGHT_SHOULDER].x * image.shape[1])
        r_shldr_y = int(lm.landmark[self.lmPose.RIGHT_SHOULDER].y * image.shape[0])

        # Left ear.
        l_ear_x = int(lm.landmark[self.lmPose.LEFT_EAR].x * image.shape[1])
        l_ear_y = int(lm.landmark[self.lmPose.LEFT_EAR].y * image.shape[0])

        # Left hip.
        l_hip_x = int(lm.landmark[self.lmPose.LEFT_HIP].x * image.shape[1])
        l_hip_y = int(lm.landmark[self.lmPose.LEFT_HIP].y * image.shape[0])

        # Calculate distance between left shoulder and right shoulder points.
        offset = self.findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

        # Calculate angles.
        neck_inclination = self.findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = self.findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

        return offset, neck_inclination, torso_inclination, l_shldr_x, r_shldr_x

def draw_text_with_background(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.9, color=(0, 255, 0), thickness=2, bg_color=(0, 0, 0)):
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    x, y = position

    # Draw background box
    cv2.rectangle(frame, (x - 10, y - text_height - 10), (x + text_width + 10, y + 10), bg_color, -1)

    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

def show_bad_posture_popup(message):
    global popup
    if 'popup' in globals() and popup.winfo_exists():
        return  # Do not create a new pop-up if one already exists
    popup = tk.Toplevel(root)
    popup.geometry("300x100+500+200")  # Center the popup on the screen
    popup.title("Posture Alert")
    label = tk.Label(popup, text=message, font=("Arial", 14))
    label.pack(pady=20)
    

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(_file_))

def play_alert_sound():
    sound_path = os.path.join(script_dir, 'data', 'sound.mp3')
    pygame.mixer.music.load(sound_path)
    pygame.mixer.music.play(-1)
def stop_alert_sound():
    pygame.mixer.music.stop()
    
# Initialize Tkinter root
root = tk.Tk()
root.withdraw()  # Hide the main window

# Initialize the model
model = PostureAnalysisModel()

# Initialize pygame mixer
pygame.mixer.init()

# Start video capture
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

bad_posture_shown = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    output = model.forward(frame)
    if output:
        offset, neck_inclination, torso_inclination, l_shldr_x, r_shldr_x = output

        # Determine posture status
        if neck_inclination < 40 and torso_inclination < 10:
            draw_text_with_background(frame, 'Good Posture', (10, 60), color=(0, 255, 0))
            if bad_posture_shown:
                bad_posture_shown = False
                if 'popup' in globals() and popup.winfo_exists():
                    popup.destroy()
                stop_alert_sound()
        else:
            if torso_inclination >= 10:
                message = "Straighten Your Back"
            elif torso_inclination >= -10:
                message = "Lean Forward and Straighten Your Back"
                
                    

            draw_text_with_background(frame, 'Bad Posture', (10, 60), color=(0, 0, 255))
            if not bad_posture_shown:
                bad_posture_shown = True
                root.after(0, lambda: show_bad_posture_popup(message))
                threading.Thread(target=play_alert_sound).start()

    # Display the frame
    cv2.imshow('Posture Analysis', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update the Tkinter root
    root.update()

# Release resources
cap.release()
cv2.destroyAllWindows()
root.destroy()
pygame.mixer.quit()


