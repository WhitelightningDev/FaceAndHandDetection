# Real-time Face and Hand Detection with Liveness Check

This project implements a real-time face and hand detection system using OpenCV, Mediapipe, and a pre-trained DNN face detection model. It includes functionalities like drawing facial landmarks, detecting hands, and performing basic liveness detection. The app uses a webcam to capture live video and provides on-screen feedback about detected faces and hands.

## Features:
- **Real-time Face Detection** using DNN-based model (`res10_300x300_ssd_iter_140000.caffemodel`).
- **Facial Landmarks Drawing** using Mediapipe.
- **Hand Detection** using Mediapipe.
- **Basic Liveness Check** by detecting face presence and changes.
- **Face Bounding Box** drawn around detected faces.
- **Real-time Display** showing video feed with detected features.

## Prerequisites:
Before you begin, ensure you have the following installed:

- Python 3.x
- `opencv-python`
- `mediapipe`
- `numpy`

### Installing Dependencies:
You can install the necessary dependencies using pip:

```bash
pip install opencv-python mediapipe numpy
```

Additionally, you will need the following two files in your project directory for face detection to work:

- `deploy.prototxt` (Caffe model configuration)
- `res10_300x300_ssd_iter_140000.caffemodel` (Pre-trained face detection model)

These files can be downloaded from the [OpenCV GitHub repository](https://github.com/opencv/opencv/tree/master/samples/dnn).

## Setup:
1. Clone or download the repository.
2. Place the `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` files in the project directory.
3. Run the Python script:

```bash
python face_hand_detection.py
```

## How It Works:
1. The webcam feed is captured using OpenCV (`cv2.VideoCapture`).
2. **Face Detection**: OpenCV DNN model (`res10_300x300_ssd_iter_140000.caffemodel`) detects faces in the video frame.
3. **Facial Landmarks**: Mediapipe FaceMesh model is used to detect and display facial landmarks on detected faces.
4. **Hand Detection**: Mediapipe Hands model detects hand landmarks and displays them on the screen.
5. **Liveness Detection**: The app performs a basic liveness check by verifying face presence in the frame.

## Code Explanation:
- **Face Detection**: Uses a deep learning-based face detection model for better accuracy compared to traditional methods (e.g., Haar Cascades).
- **Mediapipe for Face and Hand Landmark Detection**: Draws landmarks on faces and hands for visual feedback.
- **Basic Liveness Check**: The application checks if faces are detected and provides a message on-screen.

### Key Functions:
- `detect_faces(frame)`: Detects faces in the given frame using DNN-based face detection.
- `draw_landmarks(frame)`: Draws facial landmarks using Mediapipe FaceMesh.
- `detect_hands(frame)`: Detects hands in the given frame using Mediapipe Hands.
- `check_liveness(frame, faces)`: Checks for human presence by detecting faces and drawing a corresponding label.

## Usage:
- Upon running the script, the application will display the webcam feed.
- Faces will be detected and outlined with bounding boxes, while facial landmarks will be drawn on each detected face.
- Hand landmarks will be drawn if any hands are detected in the frame.
- The app will show a message on-screen if a face is detected (simple liveness check).

Press **'q'** to exit the application.

## Example Output:
A real-time video feed will display:
- **Bounding Boxes** around detected faces.
- **Green Circles** marking facial landmarks.
- **Red Circles** marking hand landmarks.

## Known Issues:
- The accuracy of face detection may be affected by lighting conditions or multiple faces in the frame.
- The application performs a basic liveness check, but it can be further improved by implementing more sophisticated techniques (e.g., blink detection).

## License:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
