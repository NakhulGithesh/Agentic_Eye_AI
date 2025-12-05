from camera_config import EXTERNAL_CAMERA_ID
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

# Disable PyAutoGUI fail-safe
pyautogui.FAILSAFE = False

class RelativeEyeController:
    def __init__(self):
        # MediaPipe initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        
        # Screen properties
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Load calibration
        self.calibration_data = None
        self.rbf_interpolator_x = None
        self.rbf_interpolator_y = None
        self.load_calibration()
        
        # Tracking buffers
        self.gaze_buffer = deque(maxlen=3)
        self.stable_gaze_buffer = deque(maxlen=2)
        self.last_mouse_pos = None
        self.movement_threshold = 8
        self.smoothing_factor = 0.15
        
        # Blink detection
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
        
        # Eye landmark indices for relative positioning
        self.left_eye_landmarks = {
            'outer_corner': 33,    # Left outer corner
            'inner_corner': 133,   # Left inner corner
            'top': 159,           # Left eye top
            'bottom': 145,        # Left eye bottom
            'iris': [474, 475, 476, 477]  # Left iris
        }
        
        self.right_eye_landmarks = {
            'outer_corner': 362,   # Right outer corner  
            'inner_corner': 263,   # Right inner corner
            'top': 386,           # Right eye top
            'bottom': 374,        # Right eye bottom
            'iris': [469, 470, 471, 472]  # Right iris
        }
        
        # Setup keyboard listener
        self.setup_keyboard_listener()
        
        print("Relative Eye Controller Initialized!")
        print("Features:")
        print("+ Relative iris positioning (eye socket-based)")
        print("+ Flipped X-coordinate for natural mapping")
        print("+ Person-independent tracking")
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
        print("Starting recalibration with relative positioning...")
        try:
            from relative_calibration import RelativeEyeTracker
            tracker = RelativeEyeTracker()
            calibration_result = tracker.enhanced_calibration()
            if calibration_result:
                tracker.save_calibration()
                self.load_calibration()
                print("[+] Relative calibration completed!")
            else:
                print("[-] Recalibration failed.")
        except Exception as e:
            print(f"Recalibration error: {e}")
    
    def calculate_relative_iris_position(self, landmarks):
        """Calculate iris position relative to eye socket landmarks"""
        if not landmarks:
            return None
        
        # Get left eye landmarks
        left_outer = landmarks.landmark[self.left_eye_landmarks['outer_corner']]
        left_inner = landmarks.landmark[self.left_eye_landmarks['inner_corner']]
        left_top = landmarks.landmark[self.left_eye_landmarks['top']]
        left_bottom = landmarks.landmark[self.left_eye_landmarks['bottom']]
        
        # Get right eye landmarks
        right_outer = landmarks.landmark[self.right_eye_landmarks['outer_corner']]
        right_inner = landmarks.landmark[self.right_eye_landmarks['inner_corner']]
        right_top = landmarks.landmark[self.right_eye_landmarks['top']]
        right_bottom = landmarks.landmark[self.right_eye_landmarks['bottom']]
        
        # Calculate iris centers
        left_iris_points = [landmarks.landmark[i] for i in self.left_eye_landmarks['iris']]
        right_iris_points = [landmarks.landmark[i] for i in self.right_eye_landmarks['iris']]
        
        left_iris_x = sum(p.x for p in left_iris_points) / len(left_iris_points)
        left_iris_y = sum(p.y for p in left_iris_points) / len(left_iris_points)
        
        right_iris_x = sum(p.x for p in right_iris_points) / len(right_iris_points)
        right_iris_y = sum(p.y for p in right_iris_points) / len(right_iris_points)
        
        # Calculate eye socket dimensions for normalization
        left_eye_width = abs(left_outer.x - left_inner.x)
        left_eye_height = abs(left_top.y - left_bottom.y)
        
        right_eye_width = abs(right_outer.x - right_inner.x)
        right_eye_height = abs(right_top.y - right_bottom.y)
        
        # Calculate relative positions within eye sockets (0.0 to 1.0)
        # For left eye: 0.0 = outer corner, 1.0 = inner corner
        if left_eye_width > 0:
            left_relative_x = (left_iris_x - left_outer.x) / left_eye_width
        else:
            left_relative_x = 0.5
            
        if left_eye_height > 0:
            left_relative_y = (left_iris_y - left_top.y) / left_eye_height
        else:
            left_relative_y = 0.5
        
        # For right eye: 0.0 = outer corner, 1.0 = inner corner
        if right_eye_width > 0:
            right_relative_x = (right_iris_x - right_outer.x) / right_eye_width
        else:
            right_relative_x = 0.5
            
        if right_eye_height > 0:
            right_relative_y = (right_iris_y - right_top.y) / right_eye_height
        else:
            right_relative_y = 0.5
        
        # Average both eyes for final gaze direction
        avg_relative_x = (left_relative_x + right_relative_x) / 2.0
        avg_relative_y = (left_relative_y + right_relative_y) / 2.0
        
        # Use direct X coordinate for natural mapping (looking left moves cursor left)
        flipped_x = avg_relative_x
        
        # Clamp values to valid range
        flipped_x = max(0.0, min(1.0, flipped_x))
        avg_relative_y = max(0.0, min(1.0, avg_relative_y))
        
        return (flipped_x, avg_relative_y)
    
    def detect_iris_relative(self, frame):
        """Enhanced iris detection with relative positioning"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None, None
        
        landmarks = results.multi_face_landmarks[0]
        
        # Calculate relative iris position
        relative_gaze = self.calculate_relative_iris_position(landmarks)
        
        if not relative_gaze:
            return None, landmarks
        
        # Add to stability buffer
        current_gaze = relative_gaze
        self.gaze_buffer.append(current_gaze)
        
        # Apply temporal stability filtering
        if len(self.gaze_buffer) >= 2:
            gaze_array = np.array(list(self.gaze_buffer))
            stable_x = np.median(gaze_array[:, 0])
            stable_y = np.median(gaze_array[:, 1])
            
            # Micro-movement suppression
            if len(self.stable_gaze_buffer) > 0:
                last_stable = self.stable_gaze_buffer[-1]
                distance = math.sqrt((stable_x - last_stable[0])**2 + (stable_y - last_stable[1])**2)
                if distance < 0.05:  # Adjusted threshold for relative coordinates
                    stable_x = last_stable[0] * 0.6 + stable_x * 0.4
                    stable_y = last_stable[1] * 0.6 + stable_y * 0.4
            
            stable_gaze = (stable_x, stable_y)
            self.stable_gaze_buffer.append(stable_gaze)
            return stable_gaze, landmarks
        
        return current_gaze, landmarks
    
    def map_gaze_to_screen(self, relative_gaze):
        """Map relative gaze position to screen coordinates"""
        if not self.rbf_interpolator_x or not self.rbf_interpolator_y or not relative_gaze:
            # Fallback to direct mapping if no calibration
            if relative_gaze:
                pixel_x = int(relative_gaze[0] * self.screen_width)
                pixel_y = int(relative_gaze[1] * self.screen_height)
                return (pixel_x, pixel_y)
            return None
        
        try:
            screen_x = self.rbf_interpolator_x(relative_gaze[0], relative_gaze[1])
            screen_y = self.rbf_interpolator_y(relative_gaze[0], relative_gaze[1])
            
            screen_x = np.clip(screen_x, 0, 1)
            screen_y = np.clip(screen_y, 0, 1)
            
            pixel_x = int(screen_x * self.screen_width)
            pixel_y = int(screen_y * self.screen_height)
            
            return (pixel_x, pixel_y)
            
        except Exception as e:
            if self.show_debug:
                print(f"Mapping error: {e}")
            # Fallback to direct mapping
            pixel_x = int(relative_gaze[0] * self.screen_width)
            pixel_y = int(relative_gaze[1] * self.screen_height)
            return (pixel_x, pixel_y)
    
    def detect_enhanced_blink(self, landmarks):
        """Enhanced blink detection"""
        if not landmarks or self.blink_cooldown > 0:
            if self.blink_cooldown > 0:
                self.blink_cooldown -= 1
            return False
        
        # EAR-based blink detection
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
        """Calculate Eye Aspect Ratio"""
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
        """Optimized mouse movement"""
        if not screen_pos or self.paused:
            return
        
        current_mouse_pos = pyautogui.position()
        
        if self.last_mouse_pos is None:
            self.last_mouse_pos = screen_pos
            return
        
        distance = math.sqrt((screen_pos[0] - current_mouse_pos[0])**2 + 
                           (screen_pos[1] - current_mouse_pos[1])**2)
        
        if distance > self.movement_threshold:
            # Adaptive smoothing
            if distance > 150:
                smoothing = self.smoothing_factor * 0.2
            elif distance > 50:
                smoothing = self.smoothing_factor * 0.5
            else:
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
    
    def draw_relative_debug(self, frame, relative_gaze, screen_pos, landmarks):
        """Enhanced debug overlay for relative tracking"""
        if not self.show_debug:
            return frame
        
        h, w, _ = frame.shape
        
        # Draw relative iris positions
        if relative_gaze:
            # Convert relative coordinates to frame coordinates for visualization
            rel_x, rel_y = int(relative_gaze[0] * w), int(relative_gaze[1] * h)
            
            # Draw relative gaze indicator
            cv2.circle(frame, (rel_x, rel_y), 6, (0, 255, 255), -1)  # Yellow center
            cv2.circle(frame, (rel_x, rel_y), 10, (255, 255, 255), 2)  # White border
            cv2.circle(frame, (rel_x, rel_y), 14, (0, 255, 255), 1)  # Yellow outer ring
            
            # Draw crosshair
            cv2.line(frame, (rel_x - 12, rel_y), (rel_x + 12, rel_y), (0, 255, 255), 2)
            cv2.line(frame, (rel_x, rel_y - 12), (rel_x, rel_y + 12), (0, 255, 255), 2)
            
            # Display relative coordinates
            cv2.putText(frame, f"Rel: ({relative_gaze[0]:.3f}, {relative_gaze[1]:.3f})", 
                       (rel_x - 80, rel_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Draw eye socket landmarks for reference
        if landmarks:
            # Left eye landmarks
            left_landmarks = [
                self.left_eye_landmarks['outer_corner'],
                self.left_eye_landmarks['inner_corner'],
                self.left_eye_landmarks['top'],
                self.left_eye_landmarks['bottom']
            ]
            
            for idx in left_landmarks:
                point = landmarks.landmark[idx]
                x, y = int(point.x * w), int(point.y * h)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)  # Blue for left eye
            
            # Right eye landmarks
            right_landmarks = [
                self.right_eye_landmarks['outer_corner'],
                self.right_eye_landmarks['inner_corner'],
                self.right_eye_landmarks['top'],
                self.right_eye_landmarks['bottom']
            ]
            
            for idx in right_landmarks:
                point = landmarks.landmark[idx]
                x, y = int(point.x * w), int(point.y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Green for right eye
        
        # Status information
        status_color = (0, 255, 0) if not self.paused else (0, 0, 255)
        status_text = "RELATIVE ACTIVE" if not self.paused else "PAUSED"
        
        cv2.putText(frame, f"Status: {status_text}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Screen coordinates
        if screen_pos:
            cv2.putText(frame, f"Screen: ({screen_pos[0]}, {screen_pos[1]})", (10, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Mouse position
        mouse_pos = pyautogui.position()
        cv2.putText(frame, f"Mouse: ({mouse_pos.x}, {mouse_pos.y})", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Tracking mode
        cv2.putText(frame, "Mode: Relative", (10, frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Calibration status
        cal_status = "RBF Cal" if self.rbf_interpolator_x else "Direct Map"
        cal_color = (0, 255, 0) if self.rbf_interpolator_x else (255, 255, 0)
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
        """Main control loop with relative positioning"""
        print("[*] Starting Relative Eye Controller...")
        print("Using eye socket-relative positioning for person-independent tracking")
        
        cap = cv2.VideoCapture(EXTERNAL_CAMERA_ID)
        if not cap.isOpened():
            print("[-] Cannot open webcam")
            return
        
        # Optimized camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        window_name = "Relative Eye Controller"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
        
        print("[+] Relative eye controller started!")
        print("Features: Flipped X-axis, Eye socket-relative positioning")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                self.update_fps()
                
                # Use relative iris detection
                relative_gaze, landmarks = self.detect_iris_relative(frame)
                screen_pos = None
                
                if relative_gaze:
                    # Map to screen coordinates
                    screen_pos = self.map_gaze_to_screen(relative_gaze)
                    
                    if screen_pos and not self.paused:
                        # Move mouse
                        self.move_mouse_optimized(screen_pos)
                    
                    # Detect blinks
                    if self.detect_enhanced_blink(landmarks):
                        self.perform_click()
                
                # Draw debug overlay
                frame = self.draw_relative_debug(frame, relative_gaze, screen_pos, landmarks)
                
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
            print("\n[!] Shutting down relative controller...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.keyboard_listener.stop()
            print("[+] Relative Eye Controller stopped.")

def main():
    """Main function"""
    print("[*] Relative Eye-Controlled PC System")
    print("=" * 50)
    print("Features:")
    print("+ Eye socket-relative positioning")
    print("+ Flipped X-coordinate mapping") 
    print("+ Person-independent tracking")
    print("+ Works without calibration")
    print("=" * 50)
    
    try:
        controller = RelativeEyeController()
        controller.run()
    except Exception as e:
        print(f"[-] Error: {e}")
        print("Please ensure your webcam is connected.")

if __name__ == "__main__":
    main()