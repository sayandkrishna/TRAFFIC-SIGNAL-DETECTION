# Traffic Sign Detection Using YOLOv8

## Overview

This project uses the YOLOv8 model to detect and classify various traffic signs in real-time. It is designed to identify traffic signs such as speed limits, stop signs, pedestrian crossings, and warning signs from video frames or live camera feeds. The model is trained on labeled images and then deployed for real-time detection, providing feedback in the form of bounding boxes and labels around the detected signs.

## Features

- **Traffic Sign Detection**: Detects and classifies different types of traffic signs, including:
  - Speed Limit Signs (e.g., Speed Limit 30, 50)
  - Stop Signs
  - Yield Signs
  - No Entry Signs
  - Pedestrian Crossing Signs
  - Parking Signs
  - Warning Signs (e.g., curve ahead, slippery road)
  - Turn Restriction Signs (e.g., no left turn, no right turn)
- **Real-Time Detection**: Processes live video feeds or webcam input for real-time traffic sign detection.
- **Bounding Box & Labeling**: Draws bounding boxes around detected signs and labels them with their respective class (e.g., "Speed Limit 50").
- **OpenCV Integration**: Uses OpenCV for video capture and displaying real-time detection results.

## Requirements

- Python 3.x
- YOLOv8 (Ultralytics YOLOv8)
- OpenCV
- PyTorch
