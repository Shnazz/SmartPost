from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog
import os
import shutil


class PostureAnalysisUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Posture Analysis')
        self.setGeometry(100, 100, 400, 500)

        # Create and set up UI elements
        self.video_label = QLabel(self)
        self.video_label.setText('')  # Initially, the label is empty

        self.start_button = QPushButton('Start Analysis', self)
        self.start_button.clicked.connect(self.start_analysis)

        self.change_sound_button = QPushButton('Change Sound', self)
        self.change_sound_button.clicked.connect(self.change_sound)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.change_sound_button)  # Add the "Change Sound" button to the layout

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def start_analysis(self):
        self.video_label.setText('Camera Feed Started')
        from Feat1234 import run 
        run()  # Run the analysis
        self.video_label.setText('Camera Feed Stopped')  # Update the label after `run()` completes

    def change_sound(self):
        # Open a file dialog to select an MP3 file
        sound_file, _ = QFileDialog.getOpenFileName(self, "Select Sound File", "", "Audio Files (*.mp3)")
        if sound_file:
            # Define the path to the current sound file in the 'data' directory
            data_dir = 'data'
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
