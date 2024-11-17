from ultralytics import YOLO
import cv2


def run_yolo_live(model_path, source=0):
    """
    Runs YOLO object detection on a live video feed (e.g., webcam or video stream).

    Parameters:
    - model_path: Path to the trained YOLO model.
    - source: Video source (default is 0 for webcam). You can use a video file path as well.
    """
    # Load YOLO model
    model = YOLO(model_path)

    # Open video source (webcam or video file)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Unable to open video source {source}")
        return

    print("Press 'q' to exit the live feed.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended.")
            break

        # Perform detection on the current frame
        results = model.predict(frame, save=False, stream=False)

        # Annotate the frame with detection results
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    # Extract class index and confidence
                    cls_id = int(box.cls) if box.cls is not None else -1
                    confidence = box.conf if box.conf is not None else 0.0
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy)

                    # Annotate frame with bounding boxes and class labels
                    label = f"{model.names[cls_id]} {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the annotated frame
        cv2.imshow("YOLO Live Detection", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Path to your trained YOLO model
model_path = 'trafficmodel.pt'  # Replace with your YOLO model file path

# Run YOLO live on webcam (default source=0)
run_yolo_live(model_path)
