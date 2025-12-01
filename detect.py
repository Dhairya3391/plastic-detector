import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict

class PlasticBagDetector:
    def __init__(self, model_path='runs/detect/plastic_bag_detector/weights/best.pt'):
        """
        Initialize the plastic bag detector with trained YOLOv11 model
        """
        try:
            self.model = YOLO(model_path)
            print(f"Model loaded from: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to pretrained YOLOv11n for demonstration")
            self.model = YOLO('yolo11n.pt')

        # For tracking multiple bags
        self.tracked_objects = defaultdict(list)
        self.next_id = 0

        # Detection parameters
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45

    def detect_plastic_bags(self, frame):
        """
        Detect plastic bags in a frame and return annotated frame with bounding boxes
        """
        # Run YOLOv11 inference
        results = self.model(frame, conf=self.confidence_threshold, iou=self.iou_threshold)

        annotated_frame = frame.copy()
        detection_count = 0

        # Process detections
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    # Only process plastic bag detections (class 0)
                    if class_id == 0:  # plastic-bag class
                        detection_count += 1

                        # Convert to integers
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # Draw bounding box
                        color = (0, 255, 0)  # Green for plastic bags
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                        # Add label with confidence
                        label = f'Plastic Bag: {confidence:.2f}'
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

                        # Draw label background
                        cv2.rectangle(annotated_frame,
                                    (x1, y1 - label_size[1] - 10),
                                    (x1 + label_size[0], y1),
                                    color, -1)

                        # Draw label text
                        cv2.putText(annotated_frame, label,
                                  (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Add detection count overlay
        cv2.putText(annotated_frame, f'Bags detected: {detection_count}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return annotated_frame, detection_count

    def run_webcam_detection(self):
        """
        Run real-time plastic bag detection using webcam
        """
        print("Starting webcam detection...")
        print("Press 'q' to quit, 's' to save current frame")

        # Initialize webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        # Set webcam resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Detect plastic bags
            annotated_frame, detection_count = self.detect_plastic_bags(frame)

            # Calculate FPS
            fps_counter += 1
            if fps_counter >= 10:
                current_fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()

            # Add FPS overlay
            cv2.putText(annotated_frame, f'FPS: {current_fps:.1f}',
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Display the resulting frame
            cv2.imshow('Plastic Bag Detection', annotated_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f'plastic_bag_detection_{timestamp}.jpg'
                cv2.imwrite(filename, annotated_frame)
                print(f"Frame saved as {filename}")

        # Release webcam and close windows
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam detection stopped")

    def run_video_detection(self, video_path):
        """
        Run plastic bag detection on video file
        """
        print(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create output video writer
        output_path = f'output_{video_path}'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        total_detections = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Detect plastic bags
            annotated_frame, detection_count = self.detect_plastic_bags(frame)
            total_detections += detection_count

            # Write frame to output video
            out.write(annotated_frame)

            # Display progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Video processing completed!")
        print(f"Total frames: {frame_count}")
        print(f"Total plastic bag detections: {total_detections}")
        print(f"Output saved as: {output_path}")


def main():
    """
    Main function to run plastic bag detection
    """
    print("=== Plastic Bag Detection System ===")
    print("1. Train model first using: python train.py")
    print("2. Run detection after training is complete")
    print()

    detector = PlasticBagDetector()

    # Ask user for input source
    print("Choose input source:")
    print("1. Webcam (live detection)")
    print("2. Video file")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == '1':
        detector.run_webcam_detection()
    elif choice == '2':
        video_path = input("Enter video file path: ").strip()
        detector.run_video_detection(video_path)
    else:
        print("Invalid choice. Running webcam detection by default...")
        detector.run_webcam_detection()


if __name__ == "__main__":
    main()