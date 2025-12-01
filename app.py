import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import tempfile
import os
from PIL import Image
import threading

# Page configuration
st.set_page_config(
    page_title="Plastic Bag Detection System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .detection-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stats-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class PlasticBagDetectorUI:
    def __init__(self):
        self.model = None
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        self.is_running = False
        self.cap = None

    def load_model(self, model_path):
        """Load YOLOv11 model for plastic bag detection"""
        try:
            self.model = YOLO(model_path)
            return True, "Model loaded successfully!"
        except Exception as e:
            try:
                # Fallback to pretrained model
                self.model = YOLO('yolo11n.pt')
                return True, "Using pretrained YOLOv11n model (train your model for better results)"
            except Exception as e2:
                return False, f"Error loading model: {e2}"

    def detect_plastic_bags(self, frame):
        """Detect plastic bags in frame and return annotated frame"""
        if self.model is None:
            return frame, 0

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

                        # Draw bounding box (green for plastic bags)
                        color = (0, 255, 0)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                        # Add label with confidence
                        label = f'Bag: {confidence:.2f}'
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

                        # Draw label background
                        cv2.rectangle(annotated_frame,
                                    (x1, y1 - label_size[1] - 10),
                                    (x1 + label_size[0], y1),
                                    color, -1)

                        # Draw label text
                        cv2.putText(annotated_frame, label,
                                  (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return annotated_frame, detection_count

def main():
    detector = PlasticBagDetectorUI()

    # Header
    st.markdown('<h1 class="main-header">🛍️ Plastic Bag Detection System</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    st.sidebar.title("⚙️ Settings")

    # Model selection
    st.sidebar.subheader("Model Configuration")

    # Check for trained model
    model_paths = []
    if os.path.exists('runs/detect/plastic_bag_detector/weights/best.pt'):
        model_paths.append('runs/detect/plastic_bag_detector/weights/best.pt')
    model_paths.append('yolo11n.pt')  # Pretrained fallback

    model_choice = st.sidebar.selectbox(
        "Select Model",
        model_paths,
        help="Choose your trained model or use pretrained YOLOv11n"
    )

    # Load model
    success, message = detector.load_model(model_choice)
    if success:
        st.sidebar.success(message)
    else:
        st.sidebar.error(message)
        return

    # Detection parameters
    st.sidebar.subheader("Detection Parameters")
    detector.confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score for detections"
    )

    detector.iou_threshold = st.sidebar.slider(
        "IoU Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Intersection over Union threshold for NMS"
    )

    # Main content area
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("📹 Live Detection Feed")

        # Input source selection
        input_source = st.radio(
            "Select Input Source",
            ["Webcam", "Upload Image", "Upload Video"],
            horizontal=True
        )

    with col2:
        st.subheader("📊 Detection Stats")

        # Stats placeholders
        bags_detected = st.empty()
        fps_display = st.empty()
        status_display = st.empty()

        # Initialize stats
        stats_placeholder = st.container()
        with stats_placeholder:
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            st.metric("Bags Detected", "0")
            st.metric("Current FPS", "0")
            st.metric("Status", "🟢 Ready")
            st.markdown('</div>', unsafe_allow_html=True)

    # Handle different input sources
    if input_source == "Webcam":
        st.subheader("📸 Webcam Live Detection")

        if st.button("🎥 Start Webcam", type="primary"):
            # Initialize webcam
            detector.cap = cv2.VideoCapture(0)
            detector.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            detector.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not detector.cap.isOpened():
                st.error("Could not access webcam. Please check permissions.")
                return

            # Create placeholder for video feed
            video_placeholder = st.empty()

            # FPS calculation
            fps_counter = 0
            fps_start_time = time.time()

            try:
                while detector.is_running:
                    ret, frame = detector.cap.read()
                    if not ret:
                        st.error("Could not read frame from webcam")
                        break

                    # Detect plastic bags
                    annotated_frame, detection_count = detector.detect_plastic_bags(frame)

                    # Calculate FPS
                    fps_counter += 1
                    if fps_counter >= 10:
                        current_fps = fps_counter / (time.time() - fps_start_time)
                        fps_counter = 0
                        fps_start_time = time.time()
                    else:
                        current_fps = 0

                    # Convert BGR to RGB for Streamlit
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                    # Display frame
                    video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

                    # Update stats
                    with stats_placeholder:
                        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                        st.metric("Bags Detected", str(detection_count))
                        st.metric("Current FPS", f"{current_fps:.1f}")
                        st.metric("Status", "🟢 Detecting")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Small delay to prevent overwhelming the UI
                    time.sleep(0.03)

            except Exception as e:
                st.error(f"Error during webcam detection: {e}")
            finally:
                if detector.cap:
                    detector.cap.release()
                detector.is_running = False

        if st.button("⏹️ Stop Webcam"):
            detector.is_running = False
            if detector.cap:
                detector.cap.release()

    elif input_source == "Upload Image":
        st.subheader("🖼️ Image Detection")

        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp']
        )

        if uploaded_file is not None:
            # Convert uploaded file to OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            # Detect plastic bags
            annotated_image, detection_count = detector.detect_plastic_bags(image)

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Original Image**")
                st.image(image, channels="BGR", use_column_width=True)

            with col2:
                st.write("**Detection Result**")
                st.image(annotated_image, channels="BGR", use_column_width=True)

            # Update stats
            with stats_placeholder:
                st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                st.metric("Bags Detected", str(detection_count))
                st.metric("Processing Time", "Completed")
                st.metric("Status", "✅ Done")
                st.markdown('</div>', unsafe_allow_html=True)

            if detection_count > 0:
                st.success(f"🎉 Found {detection_count} plastic bag(s) in the image!")
            else:
                st.info("No plastic bags detected in this image.")

    elif input_source == "Upload Video":
        st.subheader("🎬 Video Detection")

        uploaded_video = st.file_uploader(
            "Choose a video...",
            type=['mp4', 'avi', 'mov', 'mkv']
        )

        if uploaded_video is not None:
            # Save uploaded video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name

            # Process video
            if st.button("🎥 Process Video", type="primary"):
                st.info("Processing video... This may take a while.")

                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    # Get video properties
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    # Create output video
                    output_path = 'processed_video.mp4'
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
                        annotated_frame, detection_count = detector.detect_plastic_bags(frame)
                        total_detections += detection_count

                        # Write frame
                        out.write(annotated_frame)

                        # Update progress
                        progress = frame_count / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {frame_count}/{total_frames}...")

                    cap.release()
                    out.release()

                    # Display results
                    st.success("Video processing completed!")
                    st.info(f"Total frames processed: {frame_count}")
                    st.info(f"Total plastic bags detected: {total_detections}")

                    # Provide download link
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Processed Video",
                            data=f.read(),
                            file_name="plastic_bag_detection_output.mp4",
                            mime="video/mp4"
                        )

                except Exception as e:
                    st.error(f"Error processing video: {e}")
                finally:
                    # Clean up temporary file
                    if os.path.exists(video_path):
                        os.unlink(video_path)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🛍️ Plastic Bag Detection System | Built with YOLOv11, OpenCV & Streamlit</p>
        <p>📊 Real-time detection with confidence scoring and FPS monitoring</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()