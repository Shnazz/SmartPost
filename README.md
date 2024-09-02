# CapstoneProject
![Demo-mp4](Demo.mp4)

# Posture Analysis Application

## Overview
The Posture Analysis Application is a tool designed to help office workers maintain proper posture by providing real-time feedback on their posture using computer vision and machine learning.

## Key Features

- **Real-Time Posture Analysis**
  - Utilizes Mediapipe for detecting and analyzing key body landmarks (shoulders, hips, ears).
  - Calculates metrics like neck and torso inclination to assess posture quality.

- **Visual Feedback**
  - Displays alerts directly on the video feed indicating good or poor posture.
  - Provides actionable text alerts such as "Straighten Your Back" when posture needs adjustment.

- **Sound Alerts**
  - Customizable sound notifications to alert users of poor posture.
  - Option to enable or disable sound alerts and change the alert sound.

- **User Interface**
  - Dropdown menu for selecting available cameras.
  - Settings panel to toggle sound and pop-up alerts.
  - Buttons to start and stop posture analysis with real-time updates.

- **Integration and Flexibility**
  - Modular design integrates with the `Feat1234` module for posture analysis.
  - Error handling with informative messages for a smooth user experience.

## Technical Details
- **Languages & Libraries:** Python, OpenCV, Mediapipe, PyTorch, PyQt5
- **Platform:** Desktop application for major operating systems

## Objective
To provide a user-friendly tool that encourages proper posture, improving ergonomic practices and reducing posture-related health issues among office workers.

## How do I install it?

#### 1. Setup your platform
-   Python (3.10 recommended)
-   pip

#### 2. Clone Repository
    https://github.com/Shnazz/SmartPost

#### 3. Install dependency
We highly recommend working with a `venv` to avoid issues.

```
pip install -r requirements.txt
```

## How do I use it?

1. **Launch the Application**
   - Navigate to the project directory and run the application:
     ```bash
     python main.py
     ```

2. **Select Camera**
   - Open the application and choose an available camera from the dropdown menu.

3. **Start Posture Analysis**
   - Click on the "Start Analysis" button to begin real-time posture monitoring.

4. **Change Sound Alerts**
   - Use the "Change Sound" button to select a new alert sound (MP3 file).

5. **Enable/Disable Sound Alerts**
   - Use the toggle switch to turn sound alerts on or off.

6. **View Posture Feedback**
   - The application will display real-time feedback on your posture and provide alerts if adjustments are needed.

## Troubleshooting

- **No Camera Found**
  - Ensure your camera is connected and accessible. Check the dropdown menu for available camera options.

- **Sound Alerts Not Working**
  - Verify that you have selected a valid MP3 file and that the sound is enabled in the application settings.

- **Error Messages**
  - If you encounter an error, check the console output for details and refer to the [Issues](https://github.com/Shnazz/SmartPost/issues) page for solutions.

## Contributing

We welcome contributions to improve this project. If you have suggestions or fixes, please follow these steps:

1. Fork the repository.
2. Create a new branch for your changes.
3. Make your modifications and test thoroughly.
4. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or feedback, please reach out to [shazam110606@gmail.com](mailto:shazam110606@gmail.com).
