from ultralytics import YOLO
import os


def run_yolo_detection(video_path, output_dir, model_path):
    """
    Runs YOLO object detection on a video and saves the annotated output.

    Parameters:
    - video_path: Path to the input video file.
    - output_dir: Directory to save the output video.
    - model_path: Path to the trained YOLO model.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the YOLO model
    model = YOLO(model_path)

    # Perform detection on the video and save the output
    results = model.predict(source=video_path, save=True, save_dir=output_dir)

    # Display results summary
    print("Detection completed. Output saved in:", output_dir)
    return results


# Paths for the video, output directory, and model
video_path = 'output (online-video-cutter.com).mp4'  # Replace with your video file path
output_dir = 'yolov5'  # Define the directory to save the output
model_path = 'trafficmodel.pt'  # Path to your trained YOLO model

# Run the YOLO detection
run_yolo_detection(video_path, output_dir, model_path)
