from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget


class PostureAnalysisUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Posture Analysis')
        self.setGeometry(100, 100, 800, 600)

        # Create and set up UI elements
        self.video_label = QLabel(self)
        self.video_label.setText('Camera Feed Here')
        self.video_label.resize(640, 480)

        self.start_button = QPushButton('Start Analysis', self)
        self.start_button.clicked.connect(self.start_analysis)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.start_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def start_analysis(self):
        self.video_label.setText('Camera Feed Started')
        from Feat1234 import run 
        run() 


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = PostureAnalysisUI()
    window.show()
    sys.exit(app.exec_())