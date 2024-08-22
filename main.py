from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog, QComboBox, QCheckBox
import os
import shutil
import cv2

class PostureAnalysisUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Posture Analysis')
        self.setGeometry(100, 100, 400, 550)  # Increased height to fit checkbox

        # Create and set up UI elements
        self.video_label = QLabel(self)
        self.video_label.setText('')  # Initially, the label is empty

        self.camera_selector = QComboBox(self)
        self.populate_camera_options()

        self.start_button = QPushButton('Start Analysis', self)
        self.start_button.clicked.connect(self.start_analysis)

        self.change_sound_button = QPushButton('Change Sound', self)
        self.change_sound_button.clicked.connect(self.change_sound)

        self.sound_checkbox = QCheckBox('Enable Sound', self)
        self.sound_checkbox.setChecked(True)  # Default to sound enabled

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.camera_selector)
        layout.addWidget(self.start_button)
        layout.addWidget(self.change_sound_button)  # Add the "Change Sound" button to the layout
        layout.addWidget(self.sound_checkbox)  # Add the sound checkbox to the layout

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Set the style sheet for the dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #00274D;  /* Dark blue background */
            }
            QLabel {
                color: #FFFFFF;  /* White text color */
                font-size: 16px;
            }
            QPushButton {
                background-color: #004080;  /* Darker blue button background */
                color: #FFFFFF;  /* White text color */
                font-size: 16px;
                border: none;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #003366;  /* Even darker blue on hover */
            }
            QCheckBox {
                color: #FFFFFF;  /* White text color */
                font-size: 16px;
            }
        """)

    def populate_camera_options(self):
        # Populate the dropdown with available cameras
        self.camera_selector.addItem("Select Camera")
        for i in range(10):  # Check first 10 camera indices
            cap = cv2.VideoCapture(i)
            if cap.read()[0]:
                self.camera_selector.addItem(f"Camera {i}", i)
            cap.release()

    def start_analysis(self):
        selected_camera_index = self.camera_selector.currentData()
        if selected_camera_index is None:
            self.video_label.setText('Please select a camera.')
            return

        self.video_label.setText(f'Camera Feed Started (Camera {selected_camera_index})')
        try:
            from Feat1234 import run
            sound_enabled = self.sound_checkbox.isChecked()  # Check if sound is enabled
            run(selected_camera_index, sound_enabled)  # Pass sound state to `run()`
            self.video_label.setText('Camera Feed Stopped')  # Update the label after `run()` completes
        except Exception as e:
            self.video_label.setText(f'Error during analysis: {e}')

    def change_sound(self):
        # Open a file dialog to select an MP3 file
        sound_file, _ = QFileDialog.getOpenFileName(self, "Select Sound File", "", "Audio Files (*.mp3)")
        if sound_file:
            # Define the path to the current sound file in the 'data' directory
            data_dir = 'data'
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)  # Create the directory if it does not exist
            target_file = os.path.join(data_dir, 'sound.mp3')
            
            # Replace the old sound file with the new one
            try:
                shutil.copy(sound_file, target_file)
                self.video_label.setText(f'Sound changed to: {os.path.basename(sound_file)}')
            except Exception as e:
                self.video_label.setText(f'Failed to change sound: {e}')

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = PostureAnalysisUI()
    window.show()
    sys.exit(app.exec_())
