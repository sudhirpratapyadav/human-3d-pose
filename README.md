# Human 3D Pose Estimation and Visualization

This project implements a pose tracking system using MediaPipe for real-time pose estimation. The system supports both webcam and RealSense cameras, allowing for both 2D and 3D visualization of tracked landmarks.

<p align="center"><img src=demo.gif width="60%"></p>

## Overview

The `pose_estimation.py` script serves as the main entry point for the application, initializing the camera, processing video frames, and visualizing pose landmarks. The `pose_visualiser.py` script provides a 3D visualization environment for the detected pose points and camera orientation.

## Features

- Real-time pose tracking using MediaPipe
- Support for webcam and RealSense cameras
- 2D and 3D visualization of pose landmarks
- Depth awareness when using depth-capable cameras
- Interactive visualization with mouse controls

## Installation

To run this project, ensure you have the following dependencies installed (requirements.txt). In some versions there is a backend conflict between `Pyqt5` and `opencv`, therefore i have mentioned specific versions for the same.

## Usage

To start the pose tracking application, run the following command in your terminal:
```python pose_estimation.py --cam-type <realsense|webcam> --depth --model <lite|full|heavy> --width <width> --height <height>```

**Arguments**
- `--cam-type:` Specify the type of camera to use (default is `realsense`).
- `--depth:` Enable depth processing (only applicable for depth cameras).
- `--model:` Choose the model type for pose estimation (default is `lite`).
- `--width:` Set the width of the video frame (default is `640`).
- `--height:` Set the height of the video frame (default is `480`).

**Example**
```
python pose_estimation.py --cam-type webcam --model lite --width 640 --height 480
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Acknowledgements
- This project uses MediaPipe for pose estimation.
- 3D visualization is powered by Vispy.
