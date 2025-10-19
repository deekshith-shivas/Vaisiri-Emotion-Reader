import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import tkinter.messagebox as msgbox
import os
from deepface import DeepFace
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

class VaisiriEmotionReader:
    def __init__(self):
        # Set up CustomTkinter appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # Create main window with increased size
        self.root = ctk.CTk()
        self.root.title("Vaisiri Emotion Reader - AI-Powered Real-Time Detection")
        self.root.geometry("1600x1000")
        self.root.configure(fg_color=("#F0F0F0", "#1a1a1a"))

        # Configure grid weights for enhanced layout
        self.root.grid_columnconfigure(0, weight=50)  # Camera section (50%)
        self.root.grid_columnconfigure(1, weight=50)  # Emotion section (50%)
        self.root.grid_rowconfigure(0, weight=0)      # Banner section
        self.root.grid_rowconfigure(1, weight=1)      # Main content section

        # Emotion mappings with proper neutral priority
        self.emotion_emojis = {
            'neutral': '😐', 'happy': '😀', 'sad': '😢', 
            'angry': '😠', 'surprise': '😲', 'disgust': '🤢', 
            'fear': '😨'
        }

        # Emotion colors
        self.emotion_colors = {
            'neutral': '#607D8B',    # Grey-blue (DEFAULT)
            'happy': '#4CAF50',      # Green
            'sad': '#2196F3',        # Blue
            'angry': '#F44336',      # Red
            'surprise': '#FF9800',   # Orange
            'disgust': '#795548',    # Brown
            'fear': '#9C27B0'        # Purple
        }

        # Camera settings
        self.CAMERA_WIDTH = 1800
        self.CAMERA_HEIGHT = 1300

        # Load banner and emoji images
        self.banner_image = None
        self.emoji_images = {}
        self.load_banner_image()
        self.load_emoji_images()

        # Emotion detection settings
        self.frame_count = 0
        self.skip_frames = 3  # Process every 3rd frame for      performance
        self.last_emotion_update = time.time()
        self.emotion_update_interval = 0.5  # Update every 0.5 seconds
        self.emotion_smoothing_buffer = []  # For emotion smoothing
        self.buffer_size = 3  # Keep last 3 predictions for smoothing

        # Current emotion state - START WITH NEUTRAL
        self.current_emotion = 'neutral'
        self.emotion_confidence = 0.0
        self.last_detection_time = time.time()
        self.no_face_timeout = 2.0  # Return to neutral after 2 seconds of no face

        # Initialize variables
        self.cap = None
        self.is_running = False
        self.detection_active = True

        # Face detection setup
        self.face_cascade = None
        self.setup_face_detection()

        # DeepFace model initialization flag
        self.deepface_ready = False
        self.initialize_deepface()

        self.setup_ui()
        self.setup_camera()

    def initialize_deepface(self):
        """Initialize DeepFace model in background"""
        def init_model():
            try:
                print("🤖 Initializing DeepFace emotion model...")
                # Load the model by making a dummy prediction
                dummy_img = np.zeros((48, 48, 3), dtype=np.uint8)
                DeepFace.analyze(dummy_img, actions=['emotion'], enforce_detection=False, silent=True)
                self.deepface_ready = True
                print("✅ DeepFace model initialized successfully!")
                self.update_status_safe("🤖 AI Model Ready - Detection Active")
            except Exception as e:
                print(f"❌ DeepFace initialization error: {e}")
                msgbox.showerror("AI Model Error", f"Could not initialize DeepFace model: {e}")
                self.update_status_safe("❌ AI Model Failed - Check Installation")

        # Initialize in background thread
        init_thread = threading.Thread(target=init_model, daemon=True)
        init_thread.start()

    def load_banner_image(self):
        """Load banner image from gallery folder"""
        banner_path = os.path.join("gallery", "banner.jpg")
        try:
            if os.path.exists(banner_path):
                banner_img = Image.open(banner_path)
                banner_img = banner_img.resize((700, 120), Image.Resampling.LANCZOS)
                self.banner_image = ImageTk.PhotoImage(banner_img)
                print("✅ Banner image loaded successfully!")
            else:
                print("⚠️ Banner image not found at gallery/banner.jpg")
        except Exception as e:
            print(f"⚠️ Could not load banner image: {e}")

    def setup_face_detection(self):
        """Setup face detection"""
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("✅ Face detection loaded successfully!")
        except Exception as e:
            print(f"⚠️ Face detection setup error: {e}")

    def load_emoji_images(self):
        """Load emoji images from gallery folder with fallback"""
        gallery_path = "gallery"
        emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

        for emotion in emotions:
            self.emoji_images[emotion] = None

            if os.path.exists(gallery_path):
                for ext in ['.jpg', '.png', '.jpeg']:
                    file_path = os.path.join(gallery_path, f"{emotion}{ext}")
                    if os.path.exists(file_path):
                        try:
                            img = Image.open(file_path).resize((280, 280), Image.Resampling.LANCZOS)
                            self.emoji_images[emotion] = ImageTk.PhotoImage(img)
                            break
                        except Exception as e:
                            print(f"⚠️ Could not load {file_path}: {e}")

    def setup_ui(self):
        """Setup the user interface"""
        # Banner Section
        banner_frame = ctk.CTkFrame(self.root, height=130, fg_color="transparent")
        banner_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(10, 0))
        banner_frame.grid_columnconfigure(0, weight=1)

        # Banner image display
        if self.banner_image:
            banner_label = ctk.CTkLabel(
                banner_frame,
                image=self.banner_image,
                text=""
            )
            banner_label.grid(row=0, column=0, pady=8)
        else:
            banner_label = ctk.CTkLabel(
                banner_frame,
                text="🤖 VAISIRI EMOTION READER - AI POWERED 🤖",
                font=ctk.CTkFont(family="Helvetica", size=32, weight="bold"),
                text_color=("#2E3440", "#ECEFF4")
            )
            banner_label.grid(row=0, column=0, pady=8)

        # Camera Section
        self.camera_frame = ctk.CTkFrame(
            self.root,
            corner_radius=15,
            fg_color=("#E5E9F0", "#2E3440")
        )
        self.camera_frame.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=(10, 20))

        # Camera header
        camera_header = ctk.CTkLabel(
            self.camera_frame,
            text="📹 Live Camera Feed",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("#4C566A", "#D8DEE9")
        )
        camera_header.pack(pady=(15, 10))

        # Camera display
        self.camera_label = ctk.CTkLabel(
            self.camera_frame,
            text="📹 Initializing Camera...",
            font=ctk.CTkFont(size=18),
            width=self.CAMERA_WIDTH,
            height=self.CAMERA_HEIGHT
        )
        self.camera_label.pack(expand=True, fill="both", padx=20, pady=(0, 15))

        # Emotion Section
        self.emotion_frame = ctk.CTkFrame(
            self.root,
            corner_radius=15,
            fg_color=("#E5E9F0", "#2E3440")
        )
        self.emotion_frame.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=(10, 20))

        # Emotion header
        emotion_header = ctk.CTkLabel(
            self.emotion_frame,
            text="🤖 AI Emotion Recognition",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=("#4C566A", "#D8DEE9")
        )
        emotion_header.pack(pady=(20, 15))

        # Main emotion display card
        self.emotion_card = ctk.CTkFrame(
            self.emotion_frame,
            corner_radius=25,
            fg_color=("#D8DEE9", "#3B4252"),
            border_width=4,
            border_color=("#607D8B", "#607D8B")  # Start with neutral color
        )
        self.emotion_card.pack(expand=True, fill="both", padx=25, pady=(0, 20))

        # Emoji display
        self.emoji_label = ctk.CTkLabel(
            self.emotion_card,
            text="😐",  # Start with neutral
            font=ctk.CTkFont(size=250),
            width=1200,
            height=250
        )
        self.emoji_label.pack(pady=(30, 10))

        # Emotion name label
        self.emotion_name_label = ctk.CTkLabel(
            self.emotion_card,
            text="NEUTRAL",  # Start with neutral
            font=ctk.CTkFont(family="Helvetica", size=28, weight="bold"),
            text_color=self.emotion_colors['neutral']
        )
        self.emotion_name_label.pack(pady=(5, 10))

        # Confidence label
        self.confidence_label = ctk.CTkLabel(
            self.emotion_card,
            text="Confidence: 0%",
            font=ctk.CTkFont(size=16),
            text_color=("#4C566A", "#D8DEE9")
        )
        self.confidence_label.pack(pady=(0, 15))

        # Status frame
        self.status_frame = ctk.CTkFrame(
            self.emotion_card,
            corner_radius=12,
            fg_color="transparent"
        )
        self.status_frame.pack(pady=10, fill="x", padx=20)

        # Status label
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="🤖 Initializing AI Model...",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=("#EBCB8B", "#EBCB8B")
        )
        self.status_label.pack()

        # Control button
        self.control_frame = ctk.CTkFrame(
            self.emotion_frame,
            corner_radius=15,
            fg_color="transparent"
        )
        self.control_frame.pack(fill="x", padx=25, pady=(0, 20))

        self.toggle_button = ctk.CTkButton(
            self.control_frame,
            text="⏸️ Pause Detection",
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self.toggle_detection,
            corner_radius=30,
            height=45,
            fg_color=("#5E81AC", "#5E81AC"),
            hover_color=("#4C70A0", "#4C70A0")
        )
        self.toggle_button.pack(pady=8, fill="x")

    def setup_camera(self):
        """Initialize camera"""
        try:
            for camera_index in [0, 1, 2]:
                self.cap = cv2.VideoCapture(camera_index)
                if self.cap.isOpened():
                    print(f"✅ Camera {camera_index} connected")
                    break
                self.cap.release()

            if self.cap.isOpened():
                # Optimize camera settings
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                self.is_running = True
                self.start_video_thread()
            else:
                raise Exception("Could not open camera")

        except Exception as e:
            msgbox.showinfo("Camera Info", f"Camera setup: {str(e)}")
            self.update_status_safe("🔴 Camera Error")

    def start_video_thread(self):
        """Start video processing thread"""
        self.video_thread = threading.Thread(target=self.process_video, daemon=True)
        self.video_thread.start()

    def process_video(self):
        """Process video frames and detect emotions"""
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
                frame = cv2.flip(frame, 1)  # Mirror the image

                # Emotion detection with DeepFace
                if (self.detection_active and self.deepface_ready and 
                    self.frame_count % self.skip_frames == 0):
                    current_time = time.time()
                    if current_time - self.last_emotion_update > self.emotion_update_interval:
                        self.detect_emotion_deepface(frame)
                        self.last_emotion_update = current_time

                # Check for timeout (no face detected)
                current_time = time.time()
                if current_time - self.last_detection_time > self.no_face_timeout:
                    if self.current_emotion != 'neutral':
                        self.root.after(0, self.update_emotion_display, 'neutral', 0.0)

                self.display_frame(frame)
            time.sleep(0.033)  # ~30 FPS

    def detect_emotion_deepface(self, frame):
        """Detect emotion using DeepFace with improved accuracy"""
        try:
            # Detect faces using OpenCV
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces) > 0:
                # Take the largest face
                largest_face = max(faces, key=lambda face: face[2] * face[3])
                x, y, w, h = largest_face

                # Extract face ROI with padding
                padding = 20
                y1 = max(0, y - padding)
                y2 = min(frame.shape[0], y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(frame.shape[1], x + w + padding)

                face_roi = frame[y1:y2, x1:x2]

                if face_roi.size > 0:
                    # Analyze emotion with DeepFace
                    result = DeepFace.analyze(
                        face_roi, 
                        actions=['emotion'], 
                        enforce_detection=False,
                        silent=True
                    )

                    if isinstance(result, list):
                        result = result[0]

                    # Get dominant emotion
                    emotions = result['emotion']
                    dominant_emotion = result['dominant_emotion'].lower()
                    confidence = emotions[result['dominant_emotion']]

                    # Apply emotion smoothing
                    self.emotion_smoothing_buffer.append((dominant_emotion, confidence))
                    if len(self.emotion_smoothing_buffer) > self.buffer_size:
                        self.emotion_smoothing_buffer.pop(0)

                    # Calculate smoothed emotion (most frequent in buffer)
                    if len(self.emotion_smoothing_buffer) >= 2:
                        emotion_counts = {}
                        total_confidence = 0

                        for emo, conf in self.emotion_smoothing_buffer:
                            emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
                            total_confidence += conf

                        # Get most frequent emotion
                        smoothed_emotion = max(emotion_counts, key=emotion_counts.get)
                        avg_confidence = total_confidence / len(self.emotion_smoothing_buffer)

                        # Only update if confidence is reasonable or emotion is neutral
                        if avg_confidence > 30 or smoothed_emotion == 'neutral':
                            if smoothed_emotion != self.current_emotion or avg_confidence > self.emotion_confidence + 10:
                                self.root.after(0, self.update_emotion_display, smoothed_emotion, avg_confidence)
                                self.last_detection_time = time.time()
                                print(f"🎭 Emotion detected: {smoothed_emotion.upper()} ({avg_confidence:.1f}%)")

        except Exception as e:
            print(f"❌ Emotion detection error: {e}")
            # Don't crash, just continue

    def update_emotion_display(self, emotion, confidence):
        """Update emotion display with new emotion"""
        try:
            emotion = emotion.lower()
            if emotion not in self.emotion_emojis:
                emotion = 'neutral'  # Fallback to neutral

            self.current_emotion = emotion
            self.emotion_confidence = confidence

            # Update emoji
            img = self.emoji_images.get(emotion)
            emoji_text = self.emotion_emojis.get(emotion, '😐')

            if img:
                self.emoji_label.configure(image=img, text="")
                self.emoji_label.image = img
            else:
                self.emoji_label.configure(image=None, text=emoji_text)

            # Update emotion name
            emotion_name = emotion.upper()
            emotion_color = self.emotion_colors.get(emotion, '#607D8B')

            self.emotion_name_label.configure(
                text=emotion_name, 
                text_color=emotion_color
            )

            # Update confidence
            self.confidence_label.configure(
                text=f"Confidence: {confidence:.1f}%"
            )

            # Update emotion card border
            self.emotion_card.configure(border_color=emotion_color)
        except Exception as e:
            print(f"❌ Display update error: {e}")

    def update_status_safe(self, status_text):
        """Safely update status from any thread"""
        def update():
            try:
                self.status_label.configure(text=status_text)
            except:
                pass
        self.root.after(0, update)

    def display_frame(self, frame):
        """Display camera frame"""
        try:
            # Resize frame for display
            frame_resized = cv2.resize(frame, (self.CAMERA_WIDTH, self.CAMERA_HEIGHT))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)

            self.root.after(0, self._update_camera_label_safe, img_tk)
        except Exception as e:
            print(f"❌ Display error: {e}")

    def _update_camera_label_safe(self, img_tk):
        """Safely update camera label"""
        try:
            self.camera_label.configure(image=img_tk, text="")
            self.camera_label.image = img_tk
        except:
            pass

    def toggle_detection(self):
        """Toggle emotion detection on/off"""
        self.detection_active = not self.detection_active

        if self.detection_active:
            self.toggle_button.configure(text="⏸️ Pause Detection")
            self.update_status_safe("🤖 AI Detection Active")
        else:
            self.toggle_button.configure(text="▶️ Resume Detection")
            self.update_status_safe("⏸️ Detection Paused")

    def on_closing(self):
        """Handle window closing"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

    def run(self):
        """Run the AI-powered emotion reader application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        print("🤖 Vaisiri AI Emotion Reader - DEEPFACE POWERED")
        print("✨ Features:")
        print("  • Real-time AI emotion detection")
        print("  • DeepFace neural network backend")
        print("  • 7 emotions: neutral, happy, sad, angry, surprise, disgust, fear")
        print("  • Emotion smoothing for stability")
        print("  • Confidence scoring")
        print("  • Neutral priority (default state)")
        print("  • Auto-return to neutral when no face detected")
        print(f"📹 Camera Resolution: {self.CAMERA_WIDTH}x{self.CAMERA_HEIGHT}")
        print("🎯 NEUTRAL emotion has priority - system starts and returns to neutral")
        print("🔧 Change expressions clearly for accurate detection!")

        self.root.mainloop()

if __name__ == "__main__":
    app = VaisiriEmotionReader()
    app.run()
