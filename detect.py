import cv2
import numpy as np
import time
import threading
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional
from inference_sdk import InferenceHTTPClient
from inference_sdk.http.entities import InferenceConfiguration

class PlasticBagDetectorV2:
    def __init__(self):
        """
        Initialize the plastic bag detector with Roboflow Inference client.
        Optimized for smooth camera usage using threading.
        """
        self.api_key = "BcIJpIJ9IgDbVzINZUpT"
        self.model_id = "plastic-bag-act4g-dndwo/1"
        
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=self.api_key
        )
        
        # Inference settings
        self.confidence_threshold = 0.70  # Increased default to reduce false positives
        self.iou_threshold = 0.50
        
        # Configure inference settings
        self._update_config()

        # Threading variables
        self.stop_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.pred_lock = threading.Lock()
        
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_predictions: List[Dict] = []
        self.inference_active = False
        
        # Performance stats
        self.fps = 0
        self.inference_fps = 0

    def _update_config(self):
        """Update the inference configuration."""
        config = InferenceConfiguration(
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold
        )
        self.client.configure(config)

    def _draw_predictions(self, frame: np.ndarray, predictions: List[Dict]) -> np.ndarray:
        """Overlay detections on a frame."""
        annotated = frame.copy()
        height, width = annotated.shape[:2]

        # Filter predictions by current confidence threshold again, just in case
        valid_predictions = [p for p in predictions if p["confidence"] >= self.confidence_threshold]

        for prediction in valid_predictions:
            # Check if coordinates are normalized (0-1) or absolute
            # The SDK usually returns what the model outputs. 
            # If the previous code scaled them, we should check.
            # We'll assume normalized if values are small, but safe to just use the logic from v1
            
            x = prediction["x"]
            y = prediction["y"]
            w = prediction["width"]
            h = prediction["height"]
            
            # Heuristic to check if normalized. If x < 2 (and width is large), it's likely normalized.
            if x < 2 and width > 100: 
                x *= width
                y *= height
                w *= width
                h *= height

            confidence = prediction["confidence"]
            label = prediction.get("class", "plastic-bag")

            half_w, half_h = w / 2, h / 2
            x1, y1 = int(x - half_w), int(y - half_h)
            x2, y2 = int(x + half_w), int(y + half_h)

            color = (0, 255, 0) # Green
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label_text = f"{label}: {confidence:.2f}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0] + 10, y1),
                color,
                -1
            )
            cv2.putText(
                annotated,
                label_text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )

        # Draw stats
        cv2.putText(annotated, f'FPS: {self.fps:.1f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f'Inf FPS: {self.inference_fps:.1f}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw Confidence Threshold
        cv2.putText(annotated, f'Conf Thresh: {self.confidence_threshold:.2f} (+/- to adj)', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        bag_count = len(valid_predictions)
        status_color = (0, 0, 255) if bag_count > 0 else (0, 255, 0)
        status_text = f"PLASTIC DETECTED: {bag_count}" if bag_count > 0 else "NO PLASTIC"
        
        cv2.putText(annotated, status_text, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        return annotated

    def _inference_loop(self):
        """Background thread for running inference."""
        last_inference_time = time.time()
        frame_count = 0
        
        while not self.stop_event.is_set():
            frame_to_process = None
            
            # Get latest frame safely
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()
            
            if frame_to_process is not None:
                try:
                    start_time = time.time()
                    
                    # Run inference
                    result = self.client.infer(frame_to_process, model_id=self.model_id)
                    predictions = result.get("predictions", [])
                    
                    # Update predictions safely
                    with self.pred_lock:
                        self.latest_predictions = predictions
                    
                    # Calculate Inference FPS
                    frame_count += 1
                    if time.time() - last_inference_time >= 1.0:
                        self.inference_fps = frame_count / (time.time() - last_inference_time)
                        frame_count = 0
                        last_inference_time = time.time()
                        
                except Exception as e:
                    print(f"Inference error: {e}")
                    time.sleep(0.5) # Back off on error
            else:
                time.sleep(0.01)

    def run_webcam_detection(self):
        """Run smooth webcam detection using threading."""
        print("Starting smooth webcam detection...")
        print("Controls: 'q' to quit, 's' to save frame")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Start inference thread
        self.stop_event.clear()
        inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        inference_thread.start()

        fps_start_time = time.time()
        fps_counter = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break

                # Update latest frame for inference thread
                with self.frame_lock:
                    self.latest_frame = frame

                # Get latest predictions to draw
                current_predictions = []
                with self.pred_lock:
                    current_predictions = self.latest_predictions

                # Draw predictions
                annotated_frame = self._draw_predictions(frame, current_predictions)

                # Calculate Display FPS
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    self.fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()

                cv2.imshow('Plastic Bag Detection V2', annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f'plastic_bag_v2_{timestamp}.jpg'
                    cv2.imwrite(filename, annotated_frame)
                    print(f"Frame saved as {filename}")
                elif key == ord('+') or key == ord('='):
                    self.confidence_threshold = min(1.0, self.confidence_threshold + 0.05)
                    self._update_config()
                    print(f"Confidence threshold increased to {self.confidence_threshold:.2f}")
                elif key == ord('-') or key == ord('_'):
                    self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
                    self._update_config()
                    print(f"Confidence threshold decreased to {self.confidence_threshold:.2f}")

        finally:
            self.stop_event.set()
            inference_thread.join(timeout=1.0)
            cap.release()
            cv2.destroyAllWindows()
            print("Detection stopped")

    def run_video_detection(self, video_path):
        """Run detection on video file (synchronous for accuracy)."""
        print(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        output_path = f'output_v2_{Path(video_path).name}'
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Synchronous inference for video files
            try:
                result = self.client.infer(frame, model_id=self.model_id)
                predictions = result.get("predictions", [])
                annotated = self._draw_predictions(frame, predictions)
                out.write(annotated)
            except Exception as e:
                print(f"Error processing frame: {e}")
                out.write(frame)
            
            cv2.imshow('Video Processing', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Saved to {output_path}")

    def run_image_detection(self, image_path):
        """Run detection on a single image."""
        path = Path(image_path)
        if not path.exists():
            print(f"Image not found: {path}")
            return

        image = cv2.imread(str(path))
        if image is None:
            print(f"Failed to load image")
            return

        try:
            result = self.client.infer(image, model_id=self.model_id)
            predictions = result.get("predictions", [])
            annotated = self._draw_predictions(image, predictions)
            
            output_path = path.with_stem(path.stem + "_v2_detected")
            cv2.imwrite(str(output_path), annotated)
            print(f"Saved to {output_path}")
            print(f"Found {len(predictions)} objects")
        except Exception as e:
            print(f"Error: {e}")

def main():
    print("=== Plastic Bag Detection V2 (Smooth) ===")
    detector = PlasticBagDetectorV2()
    
    print("1. Webcam (Live)")
    print("2. Video File")
    print("3. Image File")
    
    choice = input("Choice (1-3): ").strip()
    
    if choice == '1':
        detector.run_webcam_detection()
    elif choice == '2':
        path = input("Video path: ").strip()
        detector.run_video_detection(path)
    elif choice == '3':
        path = input("Image path: ").strip()
        detector.run_image_detection(path)
    else:
        detector.run_webcam_detection()

if __name__ == "__main__":
    main()
