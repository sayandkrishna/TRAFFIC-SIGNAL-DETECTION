from ultralytics import YOLO
import os
import cv2
from playsound import playsound


def run_yolo_detection_with_feedback(video_path, output_dir, model_path):
    """
    Runs YOLO object detection on a video and provides visual and audio feedback.

    Parameters:
    - video_path: Path to the input video file.
    - output_dir: Directory to save the output video.
    - model_path: Path to the trained YOLO model.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the YOLO model
    model = YOLO(model_path)

    # Open the video for reading
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create a VideoWriter object to save the output video
    out_video_path = os.path.join(output_dir, 'output_with_feedback.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, frame_rate, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection on the current frame
        results = model(frame)

        # Iterate over the detected objects in the current frame
        for result in results[0].boxes:  # Correct access to boxes
            x1, y1, x2, y2 = result.xyxy[0].tolist()  # Extract coordinates
            conf = result.conf[0].item()  # Confidence score
            cls = result.cls[0].item()  # Class index
            cls_name = model.names[int(cls)]  # Get the class name from the model
            label = f'{cls_name} ({conf:.2f})'

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Check if a traffic light is detected (assuming 'traffic light' is one of the class names)
            if cls_name.lower() == 'Green Light':
                # Provide audio feedback (alert sound)
                playsound('alert.wav')  # Replace with your audio file path

        # Write the frame with feedback (visual annotations) to the output video
        out.write(frame)

        # Display the frame with feedback in real-time
        cv2.imshow("Traffic Signal Detection", frame)

        # Exit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Detection completed. Output saved in:", output_dir)


# Paths for the video, output directory, and model
video_path = 'output (online-video-cutter.com).mp4'  # Replace with your video file path
output_dir = 'yolov5'  # Define the directory to save the output
model_path = 'trafficmodel.pt'  # Path to your trained YOLO model

# Run the YOLO detection with feedback
run_yolo_detection_with_feedback(video_path, output_dir, model_path)
