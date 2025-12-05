import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import json
import os
from pynput import keyboard
from scipy.interpolate import Rbf
import threading
from collections import deque
import math
from camera_config import EXTERNAL_CAMERA_ID

# Disable PyAutoGUI fail-safe
pyautogui.FAILSAFE = False

class OptimizedEyeController:
    def __init__(self):
        # MediaPipe initialization - optimized settings
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,  # Higher for better tracking
            min_tracking_confidence=0.8
        )
        
        # Screen properties
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Load calibration
        self.calibration_data = None
        self.rbf_interpolator_x = None
        self.rbf_interpolator_y = None
        self.load_calibration()
        
        # Enhanced tracking buffers with reduced latency
        self.gaze_buffer = deque(maxlen=3)  # Reduced for faster response
        self.stable_gaze_buffer = deque(maxlen=2)
        self.last_mouse_pos = None
        self.movement_threshold = 8  # Reduced for more responsiveness
        self.smoothing_factor = 0.15  # More aggressive smoothing
        
        # Enhanced blink detection from your system
        self.blink_buffer = deque(maxlen=10)
        self.last_blinks = deque(maxlen=3)
        self.blink_threshold = 0.23
        self.blink_cooldown = 0
        
        # Control state
        self.paused = False
        self.show_debug = True
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
        
        # Setup keyboard listener
        self.setup_keyboard_listener()
        
        print("Optimized Eye Controller Initialized!")
        print("Controls:")
        print("  SPACE - Toggle pause/resume")
        print("  D - Toggle debug display")
        print("  R - Recalibrate")
        print("  Q - Quit")
        print("  Double blink - Left click")
    
    def setup_keyboard_listener(self):
        """Setup keyboard shortcuts"""
        def on_press(key):
            try:
                if hasattr(key, 'char') and key.char:
                    if key.char.lower() == 'q':
                        return False
                    elif key.char.lower() == 'r':
                        self.recalibrate()
                    elif key.char.lower() == 'd':
                        self.show_debug = not self.show_debug
                        print(f"Debug display: {'ON' if self.show_debug else 'OFF'}")
                elif key == keyboard.Key.space:
                    self.paused = not self.paused
                    print(f"Eye tracking: {'PAUSED' if self.paused else 'ACTIVE'}")
            except AttributeError:
                pass
            return True
        
        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.start()
    
    def load_calibration(self, filename="precise_calibration.json"):
        """Load calibration data and build interpolation model"""
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    self.calibration_data = json.load(f)
                
                gaze_points = np.array(self.calibration_data["gaze_points"])
                screen_points = np.array(self.calibration_data["screen_points"])
                
                self.rbf_interpolator_x = Rbf(
                    gaze_points[:, 0], gaze_points[:, 1], screen_points[:, 0],
                    function='thin_plate', smooth=0.001
                )
                self.rbf_interpolator_y = Rbf(
                    gaze_points[:, 0], gaze_points[:, 1], screen_points[:, 1],
                    function='thin_plate', smooth=0.001
                )
                
                print(f"[+] Calibration loaded: {len(gaze_points)} points")
                return True
            except Exception as e:
                print(f"Error loading calibration: {e}")
        else:
            print("No calibration file found. Please run calibration first.")
        return False
    
    def recalibrate(self):
        """Trigger recalibration"""
        print("Starting recalibration...")
        try:
            from enhanced_calibration import PreciseEyeTracker
            tracker = PreciseEyeTracker()
            calibration_result = tracker.enhanced_calibration()
            if calibration_result:
                tracker.save_calibration()
                self.load_calibration()
                print("[+] Recalibration completed!")
            else:
                print("✗ Recalibration failed.")
        except Exception as e:
            print(f"Recalibration error: {e}")
    
    def detect_iris_optimized(self, frame):
        """Optimized iris detection combining both approaches"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None, None
        
        landmarks = results.multi_face_landmarks[0]
        
        # Use the simpler approach from provided code but with your stability
        iris_landmarks = [474, 475, 476, 477]  # Right iris landmarks
        iris_points = [landmarks.landmark[i] for i in iris_landmarks]
        
        # Calculate precise iris center (simplified from provided code)
        iris_x = sum(p.x for p in iris_points) / len(iris_points)
        iris_y = sum(p.y for p in iris_points) / len(iris_points)
        
        # No X-axis flip needed - use direct coordinates
        
        # Your enhanced stability filtering
        current_gaze = (iris_x, iris_y)
        self.gaze_buffer.append(current_gaze)
        
        # Reduced buffer size for faster response but still stable
        if len(self.gaze_buffer) >= 2:
            gaze_array = np.array(list(self.gaze_buffer))
            stable_x = np.median(gaze_array[:, 0])
            stable_y = np.median(gaze_array[:, 1])
            
            # Micro-movement suppression (reduced threshold for responsiveness)
            if len(self.stable_gaze_buffer) > 0:
                last_stable = self.stable_gaze_buffer[-1]
                distance = math.sqrt((stable_x - last_stable[0])**2 + (stable_y - last_stable[1])**2)
                if distance < 0.003:  # Smaller threshold for more responsiveness
                    stable_x = last_stable[0] * 0.6 + stable_x * 0.4
                    stable_y = last_stable[1] * 0.6 + stable_y * 0.4
            
            stable_gaze = (stable_x, stable_y)
            self.stable_gaze_buffer.append(stable_gaze)
            return stable_gaze, landmarks
        
        return current_gaze, landmarks
    
    def map_gaze_to_screen(self, gaze_pos):
        """Enhanced gaze-to-screen mapping using your RBF system"""
        if not self.rbf_interpolator_x or not self.rbf_interpolator_y or not gaze_pos:
            return None
        
        try:
            screen_x = self.rbf_interpolator_x(gaze_pos[0], gaze_pos[1])
            screen_y = self.rbf_interpolator_y(gaze_pos[0], gaze_pos[1])
            
            screen_x = np.clip(screen_x, 0, 1)
            screen_y = np.clip(screen_y, 0, 1)
            
            pixel_x = int(screen_x * self.screen_width)
            pixel_y = int(screen_y * self.screen_height)
            
            return (pixel_x, pixel_y)
            
        except Exception as e:
            if self.show_debug:
                print(f"Mapping error: {e}")
            return None
    
    def detect_enhanced_blink(self, landmarks):
        """Enhanced blink detection from your system"""
        if not landmarks or self.blink_cooldown > 0:
            if self.blink_cooldown > 0:
                self.blink_cooldown -= 1
            return False
        
        # Your superior EAR-based blink detection
        left_eye_points = [landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
        right_eye_points = [landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
        
        left_ear = self._calculate_ear(left_eye_points)
        right_ear = self._calculate_ear(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0
        
        self.blink_buffer.append(avg_ear)
        
        if len(self.blink_buffer) < 6:
            return False
        
        current_time = time.time()
        recent_ears = list(self.blink_buffer)[-6:]
        
        min_ear = min(recent_ears)
        current_ear = recent_ears[-1]
        
        if (min_ear < self.blink_threshold and 
            current_ear > self.blink_threshold + 0.03 and
            min_ear < current_ear - 0.05):
            
            self.last_blinks.append(current_time)
            
            if len(self.last_blinks) >= 2:
                time_diff = self.last_blinks[-1] - self.last_blinks[-2]
                if 0.15 < time_diff < 0.7:
                    self.last_blinks.clear()
                    self.blink_cooldown = 15
                    return True
            
            cutoff_time = current_time - 1.0
            self.last_blinks = deque([t for t in self.last_blinks if t > cutoff_time], maxlen=3)
        
        return False
    
    def _calculate_ear(self, eye_points):
        """Your EAR calculation method"""
        v1 = self._distance(eye_points[1], eye_points[5])
        v2 = self._distance(eye_points[2], eye_points[4])
        h = self._distance(eye_points[0], eye_points[3])
        
        if h > 0:
            ear = (v1 + v2) / (2.0 * h)
        else:
            ear = 0.3
        
        return ear
    
    def _distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def move_mouse_optimized(self, screen_pos):
        """Optimized mouse movement with improved responsiveness"""
        if not screen_pos or self.paused:
            return
        
        current_mouse_pos = pyautogui.position()
        
        if self.last_mouse_pos is None:
            self.last_mouse_pos = screen_pos
            return
        
        distance = math.sqrt((screen_pos[0] - current_mouse_pos[0])**2 + 
                           (screen_pos[1] - current_mouse_pos[1])**2)
        
        # More responsive movement thresholds
        if distance > self.movement_threshold:
            # Adaptive smoothing optimized for speed and accuracy
            if distance > 150:  # Large movement - minimal smoothing
                smoothing = self.smoothing_factor * 0.2
            elif distance > 50:  # Medium movement
                smoothing = self.smoothing_factor * 0.5
            else:  # Small movement - full smoothing
                smoothing = self.smoothing_factor
            
            smooth_x = int(self.last_mouse_pos[0] * (1 - smoothing) + screen_pos[0] * smoothing)
            smooth_y = int(self.last_mouse_pos[1] * (1 - smoothing) + screen_pos[1] * smoothing)
            
            smooth_x = max(0, min(self.screen_width - 1, smooth_x))
            smooth_y = max(0, min(self.screen_height - 1, smooth_y))
            
            try:
                pyautogui.moveTo(smooth_x, smooth_y, duration=0)
                self.last_mouse_pos = (smooth_x, smooth_y)
            except Exception as e:
                if self.show_debug:
                    print(f"Mouse movement error: {e}")
    
    def perform_click(self):
        """Enhanced click performance"""
        if not self.paused:
            try:
                pyautogui.click()
                print("+ Click performed!")
            except Exception as e:
                if self.show_debug:
                    print(f"Click error: {e}")
    
    def draw_enhanced_debug(self, frame, gaze_pos, screen_pos, landmarks):
        """Enhanced debug overlay combining both approaches"""
        if not self.show_debug:
            return frame
        
        h, w, _ = frame.shape
        
        # Draw iris tracking (from provided code style but enhanced)
        if gaze_pos:
            iris_x, iris_y = int(gaze_pos[0] * w), int(gaze_pos[1] * h)
            
            # Enhanced visualization
            cv2.circle(frame, (iris_x, iris_y), 4, (0, 255, 0), -1)  # Green center
            cv2.circle(frame, (iris_x, iris_y), 8, (255, 255, 255), 2)  # White border
            cv2.circle(frame, (iris_x, iris_y), 12, (0, 255, 0), 1)  # Outer green ring
            
            # Crosshair
            cv2.line(frame, (iris_x - 10, iris_y), (iris_x + 10, iris_y), (0, 255, 0), 1)
            cv2.line(frame, (iris_x, iris_y - 10), (iris_x, iris_y + 10), (0, 255, 0), 1)
        
        # Status information (your enhanced system)
        status_color = (0, 255, 0) if not self.paused else (0, 0, 255)
        status_text = "OPTIMIZED ACTIVE" if not self.paused else "PAUSED"
        
        cv2.putText(frame, f"Status: {status_text}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Screen coordinates
        if screen_pos:
            cv2.putText(frame, f"Target: ({screen_pos[0]}, {screen_pos[1]})", (10, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Mouse position
        mouse_pos = pyautogui.position()
        cv2.putText(frame, f"Mouse: ({mouse_pos.x}, {mouse_pos.y})", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Calibration status
        cal_status = "Enhanced Cal" if self.rbf_interpolator_x else "Not Calibrated"
        cal_color = (0, 255, 0) if self.rbf_interpolator_x else (0, 0, 255)
        cv2.putText(frame, cal_status, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, cal_color, 1)
        
        return frame
    
    def update_fps(self):
        """FPS monitoring"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_timer >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_timer = current_time
    
    def run(self):
        """Optimized main control loop"""
        if not self.rbf_interpolator_x:
            print("[-] No valid calibration found!")
            print("Please run the calibration first using:")
            print("python enhanced_calibration.py")
            return
        
        print("[*] Starting Optimized Eye Controller...")
        print("Enhanced with your advanced calibration + simplified tracking!")
        
        cap = cv2.VideoCapture(EXTERNAL_CAMERA_ID)
        if not cap.isOpened():
            print("[-] Cannot open webcam")
            return
        
        # Optimized camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        window_name = "Optimized Eye Controller"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
        
        print("[+] Optimized eye controller started!")
        print("Combining best of both worlds: Your calibration + Simplified tracking")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                self.update_fps()
                
                # Use optimized iris detection
                gaze_pos, landmarks = self.detect_iris_optimized(frame)
                screen_pos = None
                
                if gaze_pos:
                    # Use your superior calibration mapping
                    screen_pos = self.map_gaze_to_screen(gaze_pos)
                    
                    if screen_pos and not self.paused:
                        # Optimized mouse movement
                        self.move_mouse_optimized(screen_pos)
                    
                    # Use your enhanced blink detection
                    if self.detect_enhanced_blink(landmarks):
                        self.perform_click()
                
                # Enhanced debug display
                frame = self.draw_enhanced_debug(frame, gaze_pos, screen_pos, landmarks)
                
                cv2.imshow(window_name, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self.paused = not self.paused
                    print(f"Eye tracking: {'PAUSED' if self.paused else 'ACTIVE'}")
                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                    print(f"Debug display: {'ON' if self.show_debug else 'OFF'}")
                elif key == ord('r'):
                    self.recalibrate()
        
        except KeyboardInterrupt:
            print("\n[!] Shutting down optimized controller...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.keyboard_listener.stop()
            print("[+] Optimized Eye Controller stopped.")

def main():
    """Main function"""
    print("[*] Optimized Eye-Controlled PC System")
    print("=" * 50)
    print("Best of both worlds:")
    print("+ Your advanced calibration system")
    print("+ Simplified responsive tracking")
    print("+ Enhanced stability algorithms")
    print("=" * 50)
    
    try:
        controller = OptimizedEyeController()
        controller.run()
    except Exception as e:
        print(f"[-] Error: {e}")
        print("Please ensure your webcam is connected and calibration is completed.")

if __name__ == "__main__":
    main()