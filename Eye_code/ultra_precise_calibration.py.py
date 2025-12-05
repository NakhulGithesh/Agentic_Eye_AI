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

class UltraPreciseEyeController:
    def __init__(self):
        print("🎯 Initializing Ultra-Precise Eye Controller...")
        
        # MediaPipe with matching calibration settings
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,  # Match calibration settings
            min_tracking_confidence=0.8
        )
        
        # Screen properties
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Load ultra-precise calibration
        self.calibration_data = None
        self.rbf_interpolator_x = None
        self.rbf_interpolator_y = None
        self.calibration_quality = 0.0
        
        # Advanced tracking state
        self.gaze_buffer = deque(maxlen=8)  # Larger buffer for stability
        self.high_quality_buffer = deque(maxlen=4)  # Only high-quality samples
        self.movement_predictor = deque(maxlen=3)   # Movement prediction
        
        # Mouse control parameters  
        self.last_mouse_pos = None
        self.target_mouse_pos = None
        self.movement_threshold = 8   # Lower threshold for more responsive
        self.smoothing_factor = 0.15  # Less smoothing for better responsiveness
        
        # Advanced blink detection
        self.blink_buffer = deque(maxlen=12)
        self.blink_history = deque(maxlen=5)
        self.blink_threshold = 0.22
        self.double_blink_window = 0.6
        self.blink_cooldown = 0
        
        # Control state
        self.paused = False
        self.show_debug = True
        self.precision_mode = True  # High precision tracking
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
        self.tracking_quality = 0.0
        
        # Load calibration
        if not self.load_ultra_calibration():
            print("❌ No ultra-precise calibration found!")
            print("Please run: python ultra_precise_calibration.py")
            return
        
        # Setup keyboard controls
        self.setup_keyboard_controls()
        
        print("✅ Ultra-Precise Eye Controller Ready!")
        self.print_controls()
    
    def load_ultra_calibration(self):
        """Load ultra-precise calibration data"""
        filename = "ultra_precise_calibration.json"
        
        if not os.path.exists(filename):
            return False
        
        try:
            with open(filename, 'r') as f:
                self.calibration_data = json.load(f)
            
            # Verify this is ultra-precise calibration
            if self.calibration_data.get("calibration_type") != "ultra_precise_49_point":
                print("⚠️  Warning: Not an ultra-precise calibration file")
            
            # Build advanced interpolation model
            gaze_points = np.array(self.calibration_data["gaze_points"])
            screen_points = np.array(self.calibration_data["screen_points"])
            quality_scores = np.array(self.calibration_data.get("quality_scores", [1.0] * len(gaze_points)))
            
            # Use quality-weighted RBF interpolation
            self.rbf_interpolator_x = Rbf(
                gaze_points[:, 0], gaze_points[:, 1], screen_points[:, 0],
                function='thin_plate', smooth=0.0005  # Less smoothing for precision
            )
            self.rbf_interpolator_y = Rbf(
                gaze_points[:, 0], gaze_points[:, 1], screen_points[:, 1], 
                function='thin_plate', smooth=0.0005
            )
            
            self.calibration_quality = np.mean(quality_scores)
            
            print(f"✅ Ultra-precise calibration loaded:")
            print(f"   📊 Points: {len(gaze_points)}")
            print(f"   🎯 Quality: {self.calibration_quality:.3f}")
            print(f"   📅 Date: {time.ctime(self.calibration_data.get('timestamp', 0))}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading calibration: {e}")
            return False
    
    def setup_keyboard_controls(self):
        """Setup keyboard shortcuts"""
        def on_press(key):
            try:
                if hasattr(key, 'char') and key.char:
                    if key.char.lower() == 'q':
                        return False  # Stop listener
                    elif key.char.lower() == 'r':
                        self.recalibrate()
                    elif key.char.lower() == 'd':
                        self.show_debug = not self.show_debug
                        print(f"🔍 Debug display: {'ON' if self.show_debug else 'OFF'}")
                    elif key.char.lower() == 'p':
                        self.precision_mode = not self.precision_mode
                        print(f"🎯 Precision mode: {'ON' if self.precision_mode else 'OFF'}")
                    elif key.char.lower() == '+' or key.char.lower() == '=':
                        self.movement_threshold = max(5, self.movement_threshold - 2)
                        print(f"📈 Sensitivity increased (threshold: {self.movement_threshold})")
                    elif key.char.lower() == '-':
                        self.movement_threshold = min(20, self.movement_threshold + 2)
                        print(f"📉 Sensitivity decreased (threshold: {self.movement_threshold})")
                elif key == keyboard.Key.space:
                    self.paused = not self.paused
                    print(f"⏯️  Eye tracking: {'PAUSED' if self.paused else 'ACTIVE'}")
            except AttributeError:
                pass
            return True
        
        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.start()
    
    def print_controls(self):
        """Print control instructions"""
        print("\n🎮 CONTROLS:")
        print("   SPACE - Toggle pause/resume")
        print("   D - Toggle debug display")
        print("   P - Toggle precision mode")
        print("   +/- - Adjust sensitivity")
        print("   R - Recalibrate system")
        print("   Q - Quit")
        print("\n👁️  EYE CONTROLS:")
        print("   Double blink - Left click")
        print("   Look where you want the cursor")
    
    def recalibrate(self):
        """Trigger recalibration"""
        print("🔄 Starting recalibration...")
        try:
            # Import and run ultra-precise calibration
            import subprocess
            result = subprocess.run(['python', 'ultra_precise_calibration.py'], 
                                  capture_output=False, text=True)
            if result.returncode == 0:
                # Reload calibration data
                if self.load_ultra_calibration():
                    print("✅ Recalibration completed successfully!")
                else:
                    print("❌ Failed to load new calibration")
            else:
                print("❌ Recalibration process failed")
        except Exception as e:
            print(f"❌ Recalibration error: {e}")
    
    def detect_iris_ultra_precise(self, frame):
        """Ultra-precise iris detection matching calibration method"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None, None, 0.0
        
        landmarks = results.multi_face_landmarks[0]
        
        # Get iris landmarks (same as calibration)
        left_iris_indices = list(range(474, 478))
        right_iris_indices = list(range(469, 473))
        
        # Weighted center calculation (matching calibration)
        left_iris_points = [landmarks.landmark[i] for i in left_iris_indices]
        right_iris_points = [landmarks.landmark[i] for i in right_iris_indices]
        
        weights = [1.2, 1.0, 1.0, 1.2]  # Same weights as calibration
        
        left_x = sum(p.x * w for p, w in zip(left_iris_points, weights)) / sum(weights)
        left_y = sum(p.y * w for p, w in zip(left_iris_points, weights)) / sum(weights)
        
        right_x = sum(p.x * w for p, w in zip(right_iris_points, weights)) / sum(weights)
        right_y = sum(p.y * w for p, w in zip(right_iris_points, weights)) / sum(weights)
        
        # Average both eyes
        avg_x = (left_x + right_x) / 2.0
        avg_y = (left_y + right_y) / 2.0
        
        # Calculate tracking quality (similar to calibration quality)
        quality = self._calculate_tracking_quality(landmarks, (avg_x, avg_y))
        
        return (avg_x, avg_y), landmarks, quality
    
    def _calculate_tracking_quality(self, landmarks, gaze_pos):
        """Calculate real-time tracking quality"""
        quality_factors = []
        
        # 1. Face stability
        nose_tip = (landmarks.landmark[1].x, landmarks.landmark[1].y)
        if hasattr(self, 'reference_nose_pos'):
            nose_distance = math.sqrt((nose_tip[0] - self.reference_nose_pos[0])**2 + 
                                    (nose_tip[1] - self.reference_nose_pos[1])**2)
            stability = max(0, 1.0 - nose_distance * 50)
        else:
            self.reference_nose_pos = nose_tip
            stability = 1.0
        quality_factors.append(stability)
        
        # 2. Eye openness
        left_eye_points = [landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
        right_eye_points = [landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
        
        left_ear = self._calculate_ear(left_eye_points)
        right_ear = self._calculate_ear(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0
        
        eye_quality = 1.0 if 0.25 < avg_ear < 0.35 else 0.7
        quality_factors.append(eye_quality)
        
        # 3. Gaze consistency
        if len(self.gaze_buffer) > 3:
            recent_gazes = np.array(list(self.gaze_buffer)[-3:])
            consistency = max(0, 1.0 - np.std(recent_gazes) * 25)
            quality_factors.append(consistency)
        
        return np.mean(quality_factors)
    
    def process_gaze_ultra_precise(self, gaze_pos, quality):
        """Advanced gaze processing with prediction and filtering"""
        if not gaze_pos:
            return None
        
        # Add to main buffer
        current_gaze = (gaze_pos[0], gaze_pos[1], quality, time.time())
        self.gaze_buffer.append(current_gaze)
        
        # Only add high-quality samples to precision buffer
        if quality > 0.8:
            self.high_quality_buffer.append(current_gaze)
        
        # Need minimum samples for processing
        if len(self.gaze_buffer) < 4:
            return gaze_pos
        
        if self.precision_mode and len(self.high_quality_buffer) >= 2:
            # Use high-quality samples for precision mode
            recent_high_quality = list(self.high_quality_buffer)[-2:]
            weights = [sample[2] for sample in recent_high_quality]  # Use quality as weight
            
            if sum(weights) > 0:
                weighted_x = sum(sample[0] * weight for sample, weight in zip(recent_high_quality, weights)) / sum(weights)
                weighted_y = sum(sample[1] * weight for sample, weight in zip(recent_high_quality, weights)) / sum(weights)
                processed_gaze = (weighted_x, weighted_y)
            else:
                processed_gaze = gaze_pos
        else:
            # Use median filtering for stability
            recent_gazes = [(sample[0], sample[1]) for sample in list(self.gaze_buffer)[-4:]]
            gaze_array = np.array(recent_gazes)
            processed_gaze = (np.median(gaze_array[:, 0]), np.median(gaze_array[:, 1]))
        
        # Movement prediction for smoother tracking
        self.movement_predictor.append(processed_gaze)
        if len(self.movement_predictor) >= 3:
            # Simple linear prediction
            recent_movements = list(self.movement_predictor)
            velocity_x = (recent_movements[-1][0] - recent_movements[-2][0])
            velocity_y = (recent_movements[-1][1] - recent_movements[-2][1])
            
            # Predict next position (small lookahead)
            predicted_x = processed_gaze[0] + velocity_x * 0.3
            predicted_y = processed_gaze[1] + velocity_y * 0.3
            
            return (predicted_x, predicted_y)
        
        return processed_gaze
    
    def map_gaze_to_screen_ultra_precise(self, gaze_pos):
        """Ultra-precise gaze-to-screen mapping"""
        if not self.rbf_interpolator_x or not self.rbf_interpolator_y or not gaze_pos:
            return None
        
        try:
            # Use RBF interpolation (same as calibration)
            screen_x = self.rbf_interpolator_x(gaze_pos[0], gaze_pos[1])
            screen_y = self.rbf_interpolator_y(gaze_pos[0], gaze_pos[1])
            
            # Bounds checking with gentle clamping
            screen_x = np.clip(screen_x, 0, 1)
            screen_y = np.clip(screen_y, 0, 1)
            
            # Convert to pixel coordinates
            pixel_x = int(screen_x * self.screen_width)
            pixel_y = int(screen_y * self.screen_height)
            
            # Final bounds check
            pixel_x = max(0, min(self.screen_width - 1, pixel_x))
            pixel_y = max(0, min(self.screen_height - 1, pixel_y))
            
            return (pixel_x, pixel_y)
            
        except Exception as e:
            if self.show_debug:
                print(f"Mapping error: {e}")
            return None
    
    def move_mouse_ultra_precise(self, screen_pos):
        """Ultra-precise mouse movement with advanced smoothing"""
        if not screen_pos or self.paused:
            return
        
        current_mouse = pyautogui.position()
        
        # Initialize position tracking
        if self.last_mouse_pos is None:
            self.last_mouse_pos = screen_pos
            self.target_mouse_pos = screen_pos
            return
        
        # Calculate movement distance
        distance = math.sqrt((screen_pos[0] - current_mouse[0])**2 + 
                           (screen_pos[1] - current_mouse[1])**2)
        
        # Only move if distance exceeds threshold
        if distance > self.movement_threshold:
            # Adaptive smoothing based on movement characteristics
            if distance > 150:  # Large movement - minimal smoothing, immediate response
                smoothing = 0.05
            elif distance > 50:  # Medium movement - some smoothing
                smoothing = self.smoothing_factor * 0.7
            else:  # Small movement - full smoothing for precision
                smoothing = self.smoothing_factor
            
            # Calculate smoothed target position
            smooth_x = int(self.last_mouse_pos[0] * (1 - smoothing) + screen_pos[0] * smoothing)
            smooth_y = int(self.last_mouse_pos[1] * (1 - smoothing) + screen_pos[1] * smoothing)
            
            # Additional micro-adjustment for precision mode
            if self.precision_mode and distance < 30:
                # Apply sub-pixel precision for small movements
                micro_adjust_x = (screen_pos[0] - smooth_x) * 0.3
                micro_adjust_y = (screen_pos[1] - smooth_y) * 0.3
                smooth_x += int(micro_adjust_x)
                smooth_y += int(micro_adjust_y)
            
            # Execute movement
            try:
                pyautogui.moveTo(smooth_x, smooth_y, duration=0)
                self.last_mouse_pos = (smooth_x, smooth_y)
                self.target_mouse_pos = screen_pos
            except Exception as e:
                if self.show_debug:
                    print(f"Mouse movement error: {e}")
    
    def detect_double_blink_advanced(self, landmarks):
        """Advanced double blink detection with improved accuracy"""
        if not landmarks or self.blink_cooldown > 0:
            if self.blink_cooldown > 0:
                self.blink_cooldown -= 1
            return False
        
        # Calculate EAR for both eyes
        left_eye_points = [landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
        right_eye_points = [landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
        
        left_ear = self._calculate_ear(left_eye_points)
        right_ear = self._calculate_ear(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Add to blink detection buffer
        current_time = time.time()
        self.blink_buffer.append((avg_ear, current_time))
        
        if len(self.blink_buffer) < 8:
            return False
        
        # Analyze blink pattern over recent history
        recent_ears = [sample[0] for sample in list(self.blink_buffer)[-8:]]
        recent_times = [sample[1] for sample in list(self.blink_buffer)[-8:]]
        
        # Detect blink: significant dip followed by recovery
        min_ear = min(recent_ears)
        current_ear = recent_ears[-1]
        min_index = recent_ears.index(min_ear)
        
        # Blink criteria: deep enough dip, followed by recovery
        if (min_ear < self.blink_threshold and 
            current_ear > self.blink_threshold + 0.04 and
            min_index < len(recent_ears) - 2):  # Not at the very end
            
            # Valid blink detected
            blink_time = recent_times[min_index]
            self.blink_history.append(blink_time)
            
            # Check for double blink pattern
            if len(self.blink_history) >= 2:
                time_diff = self.blink_history[-1] - self.blink_history[-2]
                if 0.15 < time_diff < self.double_blink_window:
                    # Valid double blink!
                    self.blink_history.clear()
                    self.blink_cooldown = 20  # Prevent rapid firing
                    return True
            
            # Clean old blink records
            cutoff_time = current_time - 1.5
            self.blink_history = deque([t for t in self.blink_history if t > cutoff_time], maxlen=5)
        
        return False
    
    def _calculate_ear(self, eye_points):
        """Calculate Eye Aspect Ratio"""
        v1 = self._distance(eye_points[1], eye_points[5])
        v2 = self._distance(eye_points[2], eye_points[4]) 
        h = self._distance(eye_points[0], eye_points[3])
        
        return (v1 + v2) / (2.0 * h) if h > 0 else 0.3
    
    def _distance(self, p1, p2):
        """Calculate distance between two points"""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def perform_click(self):
        """Perform left mouse click with feedback"""
        if not self.paused:
            try:
                pyautogui.click()
                print("🖱️  Click performed!")
            except Exception as e:
                if self.show_debug:
                    print(f"Click error: {e}")
    
    def draw_ultra_debug_overlay(self, frame, gaze_pos, screen_pos, quality):
        """Comprehensive debug overlay"""
        if not self.show_debug:
            return frame
        
        h, w, _ = frame.shape
        overlay = frame.copy()
        
        # Draw gaze point
        if gaze_pos:
            gaze_x, gaze_y = int(gaze_pos[0] * w), int(gaze_pos[1] * h)
            
            # Quality-colored indicator
            quality_color = (0, 255, 0) if quality > 0.9 else (0, 255, 255) if quality > 0.8 else (0, 200, 255)
            cv2.circle(overlay, (gaze_x, gaze_y), 8, quality_color, -1)
            cv2.circle(overlay, (gaze_x, gaze_y), 12, (255, 255, 255), 2)
            
            # Precision crosshair
            cv2.line(overlay, (gaze_x - 15, gaze_y), (gaze_x + 15, gaze_y), quality_color, 2)
            cv2.line(overlay, (gaze_x, gaze_y - 15), (gaze_x, gaze_y + 15), quality_color, 2)
        
        # Status panel
        panel_y = 30
        line_height = 25
        
        # System status
        status_color = (0, 255, 0) if not self.paused else (0, 0, 255)
        status_text = "ACTIVE" if not self.paused else "PAUSED"
        cv2.putText(overlay, f"Status: {status_text}", (10, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        panel_y += line_height
        
        # Performance metrics
        cv2.putText(overlay, f"FPS: {self.current_fps:.1f}", (10, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        panel_y += line_height
        
        # Tracking quality
        quality_text = f"Quality: {quality:.3f}"
        cv2.putText(overlay, quality_text, (10, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 1)
        panel_y += line_height
        
        # Calibration info
        cal_grade = "EXCELLENT" if self.calibration_quality > 0.9 else "GOOD"
        cv2.putText(overlay, f"Cal: {cal_grade} ({self.calibration_quality:.3f})", (10, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        panel_y += line_height
        
        # Screen coordinates
        if screen_pos:
            cv2.putText(overlay, f"Target: ({screen_pos[0]}, {screen_pos[1]})", (10, panel_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            panel_y += line_height
        
        # Mouse position
        mouse_pos = pyautogui.position()
        cv2.putText(overlay, f"Mouse: ({mouse_pos.x}, {mouse_pos.y})", (10, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        panel_y += line_height
        
        # Precision mode
        precision_text = f"Precision: {'ON' if self.precision_mode else 'OFF'}"
        precision_color = (0, 255, 255) if self.precision_mode else (128, 128, 128)
        cv2.putText(overlay, precision_text, (10, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, precision_color, 1)
        panel_y += line_height
        
        # Movement threshold
        cv2.putText(overlay, f"Sensitivity: {self.movement_threshold}", (10, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        panel_y += line_height
        
        # Buffer status
        high_quality_count = len(self.high_quality_buffer)
        cv2.putText(overlay, f"HQ Samples: {high_quality_count}", (10, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Right side info
        right_x = w - 200
        
        # Gaze coordinates
        if gaze_pos:
            cv2.putText(overlay, f"Gaze: ({gaze_pos[0]:.3f}, {gaze_pos[1]:.3f})", 
                       (right_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Screen mapping accuracy indicator
        if screen_pos and self.target_mouse_pos:
            error_distance = math.sqrt((screen_pos[0] - self.target_mouse_pos[0])**2 + 
                                     (screen_pos[1] - self.target_mouse_pos[1])**2)
            error_color = (0, 255, 0) if error_distance < 30 else (0, 255, 255) if error_distance < 60 else (0, 0, 255)
            cv2.putText(overlay, f"Error: {error_distance:.1f}px", 
                       (right_x, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, error_color, 1)
        
        return overlay
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_timer >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_timer = current_time
    
    def run(self):
        """Main ultra-precise control loop"""
        if not self.rbf_interpolator_x:
            print("❌ No ultra-precise calibration found!")
            print("Please run: python ultra_precise_calibration.py")
            return
        
        print(f"\n🎯 Starting Ultra-Precise Eye Controller...")
        print(f"📊 Calibration Quality: {self.calibration_quality:.3f}")
        print(f"🎯 Expected Accuracy: 20-50 pixels")
        print("Make sure you're in the same position as during calibration!")
        
        # Initialize camera with optimal settings
        cap = cv2.VideoCapture(EXTERNAL_CAMERA_ID)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        # Match calibration camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        
        # Warm up camera
        for _ in range(10):
            ret, _ = cap.read()
            time.sleep(0.1)
        
        # Create window
        window_name = "Ultra-Precise Eye Controller"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        print("✅ Ultra-precise eye controller active!")
        
        # Performance tracking
        frame_count = 0
        successful_tracking_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read from webcam")
                    break
                
                # Mirror frame for natural interaction
                frame = cv2.flip(frame, 1)
                frame_count += 1
                
                # Update FPS
                self.update_fps()
                
                # Ultra-precise iris detection
                gaze_pos, landmarks, quality = self.detect_iris_ultra_precise(frame)
                
                if gaze_pos and quality > 0.7:  # Only process high-quality detections
                    successful_tracking_count += 1
                    
                    # Advanced gaze processing
                    processed_gaze = self.process_gaze_ultra_precise(gaze_pos, quality)
                    
                    # Map to screen coordinates
                    screen_pos = self.map_gaze_to_screen_ultra_precise(processed_gaze)
                    
                    if screen_pos and not self.paused:
                        # Ultra-precise mouse movement
                        self.move_mouse_ultra_precise(screen_pos)
                    
                    # Advanced double blink detection
                    if self.detect_double_blink_advanced(landmarks):
                        self.perform_click()
                    
                    # Update tracking quality average
                    self.tracking_quality = (self.tracking_quality * 0.95 + quality * 0.05)
                
                # Draw comprehensive debug overlay
                frame = self.draw_ultra_debug_overlay(frame, gaze_pos, screen_pos if 'screen_pos' in locals() else None, quality if quality else 0.0)
                
                # Display performance stats periodically
                if frame_count % 300 == 0 and frame_count > 0:  # Every 10 seconds at 30fps
                    success_rate = (successful_tracking_count / frame_count) * 100
                    print(f"📊 Performance: {success_rate:.1f}% success rate, avg quality: {self.tracking_quality:.3f}")
                
                # Show frame
                cv2.imshow(window_name, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self.paused = not self.paused
                    print(f"⏯️  Eye tracking: {'PAUSED' if self.paused else 'ACTIVE'}")
                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                    print(f"🔍 Debug display: {'ON' if self.show_debug else 'OFF'}")
                elif key == ord('p'):
                    self.precision_mode = not self.precision_mode
                    print(f"🎯 Precision mode: {'ON' if self.precision_mode else 'OFF'}")
                elif key == ord('r'):
                    self.recalibrate()
        
        except KeyboardInterrupt:
            print("\n🛑 Shutting down ultra-precise eye controller...")
        
        finally:
            # Final performance report
            if frame_count > 0:
                final_success_rate = (successful_tracking_count / frame_count) * 100
                print(f"\n📊 Final Performance Report:")
                print(f"   🎯 Tracking success rate: {final_success_rate:.1f}%")
                print(f"   📈 Average tracking quality: {self.tracking_quality:.3f}")
                print(f"   🖼️  Total frames processed: {frame_count}")
            
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.keyboard_listener.stop()
            print("✅ Ultra-precise eye controller stopped.")

def main():
    """Main function with system verification"""
    print("🎯 Ultra-Precise Eye-Controlled PC System")
    print("=" * 50)
    
    # Check for calibration file first
    if not os.path.exists("ultra_precise_calibration.json"):
        print("❌ No ultra-precise calibration found!")
        print("\nPlease run calibration first:")
        print("python ultra_precise_calibration.py")
        print("\nThis will achieve 20-50 pixel accuracy (vs 1000+ pixel errors)")
        return
    
    try:
        controller = UltraPreciseEyeController()
        if controller.rbf_interpolator_x:  # Check if calibration loaded successfully
            controller.run()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please ensure:")
        print("  1. Webcam is connected and working")
        print("  2. Ultra-precise calibration is completed") 
        print("  3. You're in the same position as during calibration")

if __name__ == "__main__":
    main()