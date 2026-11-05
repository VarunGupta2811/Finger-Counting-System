import cv2
import mediapipe as mp
import time
from collections import deque
import customtkinter as ctk
from PIL import Image, ImageTk

# ============================================================
# MEDIAPIPE SETUP
# ============================================================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.5
)

detector = HandLandmarker.create_from_options(options)

# Skeletal connections between the 21 hand landmarks
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky & Palm
]

# ============================================================
# ILLUMINATION NORMALIZATION & TEMPORAL HYSTERESIS
# ============================================================
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def normalize_illumination(bgr_image):
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = _clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

_HISTORY_LEN = 5
_MIN_HITS_TO_CONFIRM = 2

_hand_history = {
    "Left": deque(maxlen=_HISTORY_LEN),
    "Right": deque(maxlen=_HISTORY_LEN),
}
_last_confirmed_count = {"Left": 0, "Right": 0}
_video_start_time = time.time()
_last_timestamp_ms = 0

def _next_timestamp_ms():
    global _last_timestamp_ms
    now_ms = int((time.time() - _video_start_time) * 1000)
    if now_ms <= _last_timestamp_ms:
        now_ms = _last_timestamp_ms + 1
    _last_timestamp_ms = now_ms
    return now_ms

def _update_hand_history(hand_label, finger_count):
    history = _hand_history[hand_label]
    history.append(finger_count)
    hits = sum(1 for c in history if c is not None)
    if hits >= _MIN_HITS_TO_CONFIRM:
        seen = [c for c in history if c is not None]
        _last_confirmed_count[hand_label] = seen[-1]
        return _last_confirmed_count[hand_label]
    return _last_confirmed_count[hand_label] if hits > 0 else 0

def _decay_hand_history(hand_label):
    _hand_history[hand_label].append(None)
    history = _hand_history[hand_label]
    hits = sum(1 for c in history if c is not None)
    if hits < _MIN_HITS_TO_CONFIRM:
        _last_confirmed_count[hand_label] = 0
    return _last_confirmed_count[hand_label]

def count_fingers(hand_landmarks, hand_label):
    count = 0
    if hand_label == "Right":
        if hand_landmarks[4].x < hand_landmarks[2].x:
            count += 1
    else:
        if hand_landmarks[4].x > hand_landmarks[2].x:
            count += 1
    if hand_landmarks[8].y < hand_landmarks[6].y:
        count += 1
    if hand_landmarks[12].y < hand_landmarks[10].y:
        count += 1
    if hand_landmarks[16].y < hand_landmarks[14].y:
        count += 1
    if hand_landmarks[20].y < hand_landmarks[18].y:
        count += 1
    return count

# ============================================================
# MODERN DESKTOP FRONTEND (CustomTkinter)
# ============================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class FingerCounterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Real-Time Finger Counting System — NIT Warangal")
        self.geometry("1150x680")
        self.minsize(1000, 620)

        self.cap = None
        self.is_running = False
        self.prev_frame_time = 0

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ---- Header ----
        self.header = ctk.CTkFrame(self, height=60, corner_radius=12)
        self.header.grid(row=0, column=0, columnspan=2, padx=16, pady=(16, 8), sticky="ew")
        
        self.title_label = ctk.CTkLabel(
            self.header,
            text="🖐️  Real-Time Finger Counting System",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.title_label.pack(side="left", padx=20, pady=12)

        self.status_badge = ctk.CTkLabel(
            self.header,
            text="● CAMERA OFF",
            text_color="#f87171",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.status_badge.pack(side="right", padx=20)

        # ---- Left Panel: Video Feed ----
        self.video_card = ctk.CTkFrame(self, corner_radius=16)
        self.video_card.grid(row=1, column=0, padx=(16, 8), pady=(8, 16), sticky="nsew")
        self.video_card.grid_rowconfigure(0, weight=1)
        self.video_card.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(
            self.video_card,
            text="Click 'Start Camera' to begin live hand tracking",
            font=ctk.CTkFont(size=16),
            text_color="#9ca3af"
        )
        self.video_label.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")

        # ---- Right Panel: Sidebar Controls & Live Metrics ----
        self.sidebar = ctk.CTkFrame(self, corner_radius=16)
        self.sidebar.grid(row=1, column=1, padx=(8, 16), pady=(8, 16), sticky="nsew")
        self.sidebar.grid_columnconfigure(0, weight=1)

        # Total Count Hero Card
        self.total_card = ctk.CTkFrame(self.sidebar, fg_color="#1e3a8a", corner_radius=14)
        self.total_card.grid(row=0, column=0, padx=16, pady=(16, 10), sticky="ew")
        ctk.CTkLabel(
            self.total_card, text="TOTAL FINGERS",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#93c5fd"
        ).pack(pady=(12, 0))
        self.total_val_label = ctk.CTkLabel(
            self.total_card, text="0",
            font=ctk.CTkFont(size=54, weight="bold"), text_color="#ffffff"
        )
        self.total_val_label.pack(pady=(0, 12))

        # Per-Hand Cards Row
        self.hands_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.hands_row.grid(row=1, column=0, padx=16, pady=6, sticky="ew")
        self.hands_row.grid_columnconfigure((0, 1), weight=1)

        # Left Hand Card
        self.left_card = ctk.CTkFrame(self.hands_row, corner_radius=12)
        self.left_card.grid(row=0, column=0, padx=(0, 6), sticky="ew")
        ctk.CTkLabel(
            self.left_card, text="LEFT HAND",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#9ca3af"
        ).pack(pady=(10, 0))
        self.left_val_label = ctk.CTkLabel(
            self.left_card, text="0",
            font=ctk.CTkFont(size=34, weight="bold"), text_color="#34d399"
        )
        self.left_val_label.pack(pady=(2, 10))

        # Right Hand Card
        self.right_card = ctk.CTkFrame(self.hands_row, corner_radius=12)
        self.right_card.grid(row=0, column=1, padx=(6, 0), sticky="ew")
        ctk.CTkLabel(
            self.right_card, text="RIGHT HAND",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#9ca3af"
        ).pack(pady=(10, 0))
        self.right_val_label = ctk.CTkLabel(
            self.right_card, text="0",
            font=ctk.CTkFont(size=34, weight="bold"), text_color="#34d399"
        )
        self.right_val_label.pack(pady=(2, 10))

        # Pipeline Settings Card
        self.settings_card = ctk.CTkFrame(self.sidebar, corner_radius=12)
        self.settings_card.grid(row=2, column=0, padx=16, pady=10, sticky="ew")
        ctk.CTkLabel(
            self.settings_card, text="PIPELINE CONTROLS",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#9ca3af"
        ).pack(anchor="w", padx=14, pady=(10, 6))

        self.clahe_switch = ctk.CTkSwitch(
            self.settings_card, text="CLAHE Illumination Norm"
        )
        self.clahe_switch.select()
        self.clahe_switch.pack(anchor="w", padx=14, pady=6)

        self.skeleton_switch = ctk.CTkSwitch(
            self.settings_card, text="Draw Hand Skeleton"
        )
        self.skeleton_switch.select()
        self.skeleton_switch.pack(anchor="w", padx=14, pady=(6, 12))

        # Start / Stop Buttons
        self.start_btn = ctk.CTkButton(
            self.sidebar, text="▶  Start Camera", height=42,
            fg_color="#10b981", hover_color="#059669",
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.toggle_camera
        )
        self.start_btn.grid(row=3, column=0, padx=16, pady=(12, 8), sticky="ew")

        self.fps_label = ctk.CTkLabel(
            self.sidebar, text="FPS: 0  |  History Buffer: 5 frames",
            font=ctk.CTkFont(size=12), text_color="#6b7280"
        )
        self.fps_label.grid(row=4, column=0, padx=16, pady=(4, 12))

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def toggle_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.status_badge.configure(text="● CAMERA ERROR", text_color="#ef4444")
                return
            self.is_running = True
            self.video_label.configure(text="")
            self.start_btn.configure(text="■  Stop Camera", fg_color="#ef4444", hover_color="#dc2626")
            self.status_badge.configure(text="● LIVE TRACKING", text_color="#34d399")
            self.prev_frame_time = time.time()
            self.update_frame()
        else:
            self.stop_camera()

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.configure(image=None, text="Camera stopped. Click 'Start Camera' to resume.")
        self.start_btn.configure(text="▶  Start Camera", fg_color="#10b981", hover_color="#059669")
        self.status_badge.configure(text="● CAMERA OFF", text_color="#f87171")
        self.left_val_label.configure(text="0")
        self.right_val_label.configure(text="0")
        self.total_val_label.configure(text="0")
        self.fps_label.configure(text="FPS: 0  |  History Buffer: 5 frames")

    def update_frame(self):
        if not self.is_running or not self.cap:
            return

        success, image = self.cap.read()
        if not success:
            self.stop_camera()
            return

        # Optional CLAHE Illumination Normalization
        if self.clahe_switch.get() == 1:
            image_for_detection = normalize_illumination(image)
        else:
            image_for_detection = image

        rgb_image = cv2.cvtColor(image_for_detection, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = detector.detect_for_video(mp_image, _next_timestamp_ms())

        left_count = 0
        right_count = 0

        if result.hand_landmarks:
            seen_labels = set()
            height, width, _ = image.shape

            for hand_index, hand_landmarks in enumerate(result.hand_landmarks):
                hand_label = result.handedness[hand_index][0].category_name

                if self.skeleton_switch.get() == 1:
                    # Draw skeletal connection lines
                    for start_idx, end_idx in HAND_CONNECTIONS:
                        x1 = int(hand_landmarks[start_idx].x * width)
                        y1 = int(hand_landmarks[start_idx].y * height)
                        x2 = int(hand_landmarks[end_idx].x * width)
                        y2 = int(hand_landmarks[end_idx].y * height)
                        cv2.line(image, (x1, y1), (x2, y2), (255, 200, 0), 2)

                    # Draw landmark joints
                    for landmark in hand_landmarks:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        cv2.circle(image, (x, y), 5, (0, 255, 128), -1)

                finger_count = count_fingers(hand_landmarks, hand_label)
                seen_labels.add(hand_label)
                smoothed_count = _update_hand_history(hand_label, finger_count)

                if hand_label == "Left":
                    left_count = smoothed_count
                elif hand_label == "Right":
                    right_count = smoothed_count

            for missing_label in ({"Left", "Right"} - seen_labels):
                decayed = _decay_hand_history(missing_label)
                if missing_label == "Left":
                    left_count = decayed
                else:
                    right_count = decayed
        else:
            left_count = _decay_hand_history("Left")
            right_count = _decay_hand_history("Right")

        total_count = left_count + right_count

        # Update UI Cards
        self.left_val_label.configure(text=str(left_count))
        self.right_val_label.configure(text=str(right_count))
        self.total_val_label.configure(text=str(total_count))

        # Calculate FPS
        now = time.time()
        fps = int(1 / max(now - self.prev_frame_time, 1e-5))
        self.prev_frame_time = now
        self.fps_label.configure(text=f"FPS: {fps}  |  History Buffer: 5 frames")

        # Resize frame nicely to fit the video card
        display_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(display_rgb)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(720, 520))
        self.video_label.configure(image=ctk_img)
        self.video_label.image = ctk_img

        self.after(15, self.update_frame)

    def on_closing(self):
        self.stop_camera()
        self.destroy()

if __name__ == "__main__":
    app = FingerCounterApp()
    app.mainloop()