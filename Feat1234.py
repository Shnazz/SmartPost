import cv2
import mediapipe as mp
import math as m
import torch
import torch.nn as nn
import pygame
import threading
import os
import time
import tkinter as tk

def run(camera_index, sound_enabled):
    class PostureAnalysisModel(nn.Module):
        def __init__(self):
            super(PostureAnalysisModel, self).__init__()
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
            self.lmPose = self.mp_pose.PoseLandmark
            self.mp_drawing = mp.solutions.drawing_utils

        def findDistance(self, x1, y1, x2, y2):
            return m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        def findAngle(self, x1, y1, x2, y2):
            theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
            return (180 / m.pi) * theta

        def forward(self, image):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            keypoints = self.pose.process(image_rgb)
            lm = keypoints.pose_landmarks

            if lm is None:
                return None

            self.mp_drawing.draw_landmarks(
                image,
                lm,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            l_shldr_x = int(lm.landmark[self.lmPose.LEFT_SHOULDER].x * image.shape[1])
            l_shldr_y = int(lm.landmark[self.lmPose.LEFT_SHOULDER].y * image.shape[0])
            r_shldr_x = int(lm.landmark[self.lmPose.RIGHT_SHOULDER].x * image.shape[1])
            r_shldr_y = int(lm.landmark[self.lmPose.RIGHT_SHOULDER].y * image.shape[0])
            l_ear_x = int(lm.landmark[self.lmPose.LEFT_EAR].x * image.shape[1])
            l_ear_y = int(lm.landmark[self.lmPose.LEFT_EAR].y * image.shape[0])
            l_hip_x = int(lm.landmark[self.lmPose.LEFT_HIP].x * image.shape[1])
            l_hip_y = int(lm.landmark[self.lmPose.LEFT_HIP].y * image.shape[0])

            offset = self.findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
            neck_inclination = self.findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
            torso_inclination = self.findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

            return offset, neck_inclination, torso_inclination

    def draw_text_with_background(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.9, color=(0, 255, 0), thickness=2, bg_color=(0, 0, 0)):
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size
        x, y = position
        cv2.rectangle(frame, (x - 10, y - text_height - 10), (x + text_width + 10, y + 10), bg_color, -1)
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

    def show_bad_posture_popup(message):
    # Define global popup and create new popup
        global popup
        if 'popup' in globals() and popup.winfo_exists():
            return
        popup = tk.Toplevel(root)
        popup.title("Posture Alert")

    # Ensure the popup is always on top
        popup.attributes("-topmost", True)

    # Get screen width and height
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

    # Define the width and height of the popup
        popup_width = 610
        popup_height = 83

    # Calculate position to place the popup at the bottom right corner
        x_position = screen_width - popup_width - 10  # 10 pixels from the right edge
        y_position = screen_height - popup_height - 80  # Adjusted to be 40 pixels from the bottom edge

    # Set the geometry of the popup window
        popup.geometry(f"{popup_width}x{popup_height}+{x_position}+{y_position}")

    # Configure the window to be non-resizable
        popup.resizable(False, False)

        label = tk.Label(popup, text=message, font=("Arial", 14))
        label.pack(pady=20)

    # Automatically destroy popup after 5 seconds
        popup.after(5000, popup.destroy)


    def play_alert_sound():
        if not sound_enabled:
            return  # Do not play sound if disabled
        sound_path = os.path.join(script_dir, 'data', 'sound.mp3')
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play(-1)

    def stop_alert_sound():
        pygame.mixer.music.stop()

    def check_posture(offset, neck_inclination, torso_inclination):
        if neck_inclination < 40 and torso_inclination < 10:
            if 'popup' in globals() and popup.winfo_exists():
                popup.destroy()
            stop_alert_sound()
            return False, None
        else:
            if torso_inclination >= 10:
                message = "Straighten Your Back"
            elif torso_inclination >= -10:
                message = "Lean Back and Straighten Your Back"
            else:
                message = "Adjust Your Posture"
            
            return True, message

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = tk.Tk()
    root.withdraw()
    model = PostureAnalysisModel()
    pygame.mixer.init()
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    bad_posture_shown = False
    bad_posture_timer = None
    popup_delay = 2  # Time in seconds before the popup appears

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output = model.forward(frame)
        if output:
            offset, neck_inclination, torso_inclination = output

            is_bad_posture, message = check_posture(offset, neck_inclination, torso_inclination)

            if is_bad_posture:
                if not bad_posture_shown:
                    if bad_posture_timer is None:
                        bad_posture_timer = time.time()
                    elif time.time() - bad_posture_timer >= popup_delay:
                        bad_posture_shown = True
                        threading.Thread(target=play_alert_sound).start()  # Play sound in a separate thread
                        show_bad_posture_popup(message)  # Show the message using Tkinter
                draw_text_with_background(frame, 'Bad Posture', (10, 60), color=(0, 0, 255))
            else:
                bad_posture_shown = False
                bad_posture_timer = None
                draw_text_with_background(frame, 'Good Posture', (10, 60), color=(0, 255, 0))

        draw_text_with_background(frame, 'Press "q" to Quit', (frame.shape[1] - 300, 30), color=(255, 255, 255))

        cv2.imshow('Posture Analysis', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        root.update()  # Update Tkinter GUI

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
