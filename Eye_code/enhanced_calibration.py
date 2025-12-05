from camera_config import EXTERNAL_CAMERA_ID
import cv2
import mediapipe as mp
import numpy as np
import json
import os
import pyautogui
from scipy.interpolate import griddata, Rbf
import time
import math
from collections import deque
import threading

class PreciseEyeTracker:
    def __init__(self):
        # MediaPipe initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,  # Higher confidence for better tracking
            min_tracking_confidence=0.7
        )
        
        # Screen properties
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False
        
        # Enhanced calibration points (25 points for better accuracy)
        self.calibration_points = self._generate_calibration_grid()
        
        # Tracking buffers for stability
        self.gaze_buffer = deque(maxlen=8)  # Increased buffer size
        self.stable_gaze_buffer = deque(maxlen=3)
        
        # Calibration data
        self.calibration_data = None
        self.rbf_interpolator_x = None
        self.rbf_interpolator_y = None
        
        # Blink detection
        self.blink_buffer = deque(maxlen=10)
        self.last_blinks = deque(maxlen=3)
        self.blink_threshold = 0.23
        
        # Mouse control
        self.last_mouse_pos = None
        self.movement_threshold = 15  # Minimum movement in pixels
        self.smoothing_factor = 0.3  # Reduced for more responsive movement
        
    def _generate_calibration_grid(self):
        """Generate a comprehensive 25-point calibration grid"""
        points = []
        # 5x5 grid with additional edge points
        for y in [0.05, 0.25, 0.5, 0.75, 0.95]:
            for x in [0.05, 0.25, 0.5, 0.75, 0.95]:
                points.append((x, y))
        return points
    
    def detect_iris_precise(self, frame):
        """Enhanced iris detection with better stability"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None, None
        
        landmarks = results.multi_face_landmarks[0]
        
        # Get iris landmarks with higher precision
        left_iris_indices = list(range(474, 478))
        right_iris_indices = list(range(469, 473))
        
        # Calculate iris centers with sub-pixel accuracy
        left_iris_points = [landmarks.landmark[i] for i in left_iris_indices]
        right_iris_points = [landmarks.landmark[i] for i in right_iris_indices]
        
        # Use weighted average for better center calculation
        left_x = sum(p.x for p in left_iris_points) / len(left_iris_points)
        left_y = sum(p.y for p in left_iris_points) / len(left_iris_points)
        
        right_x = sum(p.x for p in right_iris_points) / len(right_iris_points)
        right_y = sum(p.y for p in right_iris_points) / len(right_iris_points)
        
        # Average both eyes
        avg_x = (left_x + right_x) / 2.0
        avg_y = (left_y + right_y) / 2.0
        
        # Add to buffer for temporal stability
        current_gaze = (avg_x, avg_y)
        self.gaze_buffer.append(current_gaze)
        
        # Return stabilized gaze using median filtering
        if len(self.gaze_buffer) >= 5:
            gaze_array = np.array(list(self.gaze_buffer))
            stable_x = np.median(gaze_array[:, 0])
            stable_y = np.median(gaze_array[:, 1])
            
            # Further stabilization for very small movements
            if len(self.stable_gaze_buffer) > 0:
                last_stable = self.stable_gaze_buffer[-1]
                distance = math.sqrt((stable_x - last_stable[0])**2 + (stable_y - last_stable[1])**2)
                if distance < 0.008:  # Very small movement threshold
                    stable_x = last_stable[0] * 0.7 + stable_x * 0.3
                    stable_y = last_stable[1] * 0.7 + stable_y * 0.3
            
            stable_gaze = (stable_x, stable_y)
            self.stable_gaze_buffer.append(stable_gaze)
            return stable_gaze, landmarks
        
        return current_gaze, landmarks
    
    def enhanced_calibration(self):
        """Enhanced calibration with better data collection"""
        print("Starting Enhanced 25-Point Calibration...")
        
        cap = cv2.VideoCapture(EXTERNAL_CAMERA_ID)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        calibration_data = {
            "screen_points": [],
            "gaze_points": [],
            "quality_scores": [],
            "timestamp": time.time()
        }
        
        # Create fullscreen calibration window
        cv2.namedWindow("Enhanced Calibration", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Enhanced Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        blank_img = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        # Show instructions
        self._show_calibration_instructions(blank_img)
        
        print(f"Calibrating {len(self.calibration_points)} points...")
        
        for point_idx, (x_pct, y_pct) in enumerate(self.calibration_points):
            x_pos = int(x_pct * self.screen_width)
            y_pos = int(y_pct * self.screen_height)
            
            print(f"Calibrating point {point_idx + 1}/{len(self.calibration_points)}")
            
            # Collect high-quality samples
            gaze_samples = []
            quality_scores = []
            calibration_duration = 4.0  # Longer duration for better data
            start_time = time.time()
            
            # Pre-fixation period
            pre_fixation_time = 1.0
            while time.time() - start_time < pre_fixation_time:
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                frame = cv2.flip(frame, 1)
                self._show_calibration_point(blank_img, x_pos, y_pos, point_idx, 0, "Get Ready...")
                
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to cancel
                    cap.release()
                    cv2.destroyAllWindows()
                    return None
            
            # Data collection period
            actual_start = time.time()
            while time.time() - actual_start < calibration_duration:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                gaze_pos, landmarks = self.detect_iris_precise(frame)
                
                if gaze_pos and landmarks:
                    # Quality check based on face detection confidence
                    quality_score = self._calculate_gaze_quality(landmarks, gaze_pos)
                    if quality_score > 0.7:  # Only accept high-quality samples
                        gaze_samples.append(gaze_pos)
                        quality_scores.append(quality_score)
                
                # Show progress
                progress = (time.time() - actual_start) / calibration_duration
                status = f"Collecting... ({len(gaze_samples)} samples)"
                self._show_calibration_point(blank_img, x_pos, y_pos, point_idx, progress, status)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    return None
            
            # Process collected samples with outlier rejection
            if len(gaze_samples) >= 20:  # Require minimum samples
                gaze_array = np.array(gaze_samples)
                quality_array = np.array(quality_scores)
                
                # Remove outliers using statistical method
                mean_gaze = np.mean(gaze_array, axis=0)
                distances = np.sqrt(np.sum((gaze_array - mean_gaze)**2, axis=1))
                distance_threshold = np.percentile(distances, 75)  # Keep 75% of data
                
                valid_indices = distances <= distance_threshold
                filtered_gaze = gaze_array[valid_indices]
                filtered_quality = quality_array[valid_indices]
                
                if len(filtered_gaze) >= 10:  # Still have enough data
                    # Weighted average based on quality scores
                    weights = filtered_quality / np.sum(filtered_quality)
                    final_gaze_x = np.average(filtered_gaze[:, 0], weights=weights)
                    final_gaze_y = np.average(filtered_gaze[:, 1], weights=weights)
                    
                    calibration_data["screen_points"].append([x_pct, y_pct])
                    calibration_data["gaze_points"].append([final_gaze_x, final_gaze_y])
                    calibration_data["quality_scores"].append(np.mean(filtered_quality))
                    
                    print(f"  + Point {point_idx + 1}: {len(filtered_gaze)} quality samples")
                else:
                    print(f"  - Point {point_idx + 1}: Not enough quality data after filtering")
            else:
                print(f"  - Point {point_idx + 1}: Insufficient samples collected")
        
        cap.release()
        
        # Build enhanced interpolation model
        if len(calibration_data["gaze_points"]) >= 15:  # Need minimum points for good interpolation
            self._build_interpolation_model(calibration_data)
            self._show_calibration_complete(blank_img, len(calibration_data["gaze_points"]))
            self.calibration_data = calibration_data
            return calibration_data
        else:
            self._show_calibration_failed(blank_img)
            return None
    
    def _calculate_gaze_quality(self, landmarks, gaze_pos):
        """Calculate quality score for gaze sample"""
        # Check face stability (head movement)
        face_landmarks = [landmarks.landmark[i] for i in [1, 33, 61, 291, 199]]  # Key face points
        face_points = [(l.x, l.y) for l in face_landmarks]
        
        # Calculate face bounding box stability
        face_array = np.array(face_points)
        face_std = np.std(face_array, axis=0)
        stability_score = max(0, 1.0 - np.mean(face_std) * 10)
        
        # Check gaze consistency with recent samples
        consistency_score = 1.0
        if len(self.gaze_buffer) > 3:
            recent_gazes = np.array(list(self.gaze_buffer)[-3:])
            gaze_std = np.std(recent_gazes, axis=0)
            consistency_score = max(0, 1.0 - np.mean(gaze_std) * 20)
        
        return (stability_score + consistency_score) / 2.0
    
    def _build_interpolation_model(self, calibration_data):
        """Build RBF interpolation model for precise mapping"""
        gaze_points = np.array(calibration_data["gaze_points"])
        screen_points = np.array(calibration_data["screen_points"])
        quality_weights = np.array(calibration_data["quality_scores"])
        
        # Build separate RBF interpolators for x and y coordinates
        self.rbf_interpolator_x = Rbf(
            gaze_points[:, 0], gaze_points[:, 1], screen_points[:, 0],
            function='thin_plate', smooth=0.001
        )
        self.rbf_interpolator_y = Rbf(
            gaze_points[:, 0], gaze_points[:, 1], screen_points[:, 1],
            function='thin_plate', smooth=0.001
        )
        
        print("Enhanced interpolation model built successfully!")
    
    def map_gaze_to_screen_precise(self, gaze_pos):
        """Precise gaze-to-screen mapping using RBF interpolation"""
        if not self.rbf_interpolator_x or not self.rbf_interpolator_y or not gaze_pos:
            return None
        
        try:
            # Use RBF interpolation for smooth mapping
            screen_x = self.rbf_interpolator_x(gaze_pos[0], gaze_pos[1])
            screen_y = self.rbf_interpolator_y(gaze_pos[0], gaze_pos[1])
            
            # Clamp to screen bounds
            screen_x = np.clip(screen_x, 0, 1)
            screen_y = np.clip(screen_y, 0, 1)
            
            # Convert to pixel coordinates
            pixel_x = int(screen_x * self.screen_width)
            pixel_y = int(screen_y * self.screen_height)
            
            return (pixel_x, pixel_y)
            
        except Exception as e:
            print(f"Mapping error: {e}")
            return None
    
    def detect_double_blink(self, landmarks):
        """Enhanced double blink detection"""
        if not landmarks:
            return False
        
        # Calculate Eye Aspect Ratio (EAR) for both eyes
        left_eye_points = [landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
        right_eye_points = [landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
        
        left_ear = self._calculate_ear(left_eye_points)
        right_ear = self._calculate_ear(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Add to blink buffer
        self.blink_buffer.append(avg_ear)
        
        if len(self.blink_buffer) < 5:
            return False
        
        # Detect blink pattern
        current_time = time.time()
        recent_ears = list(self.blink_buffer)[-5:]
        
        # Look for blink pattern (low EAR followed by high EAR)
        if min(recent_ears) < self.blink_threshold and recent_ears[-1] > self.blink_threshold + 0.02:
            # Blink detected
            self.last_blinks.append(current_time)
            
            # Check for double blink (two blinks within 0.8 seconds)
            if len(self.last_blinks) >= 2:
                time_diff = self.last_blinks[-1] - self.last_blinks[-2]
                if 0.2 < time_diff < 0.8:  # Valid double blink timing
                    self.last_blinks.clear()  # Reset to prevent multiple triggers
                    return True
        
        return False
    
    def _calculate_ear(self, eye_points):
        """Calculate Eye Aspect Ratio"""
        # Vertical distances
        v1 = self._distance(eye_points[1], eye_points[5])
        v2 = self._distance(eye_points[2], eye_points[4])
        
        # Horizontal distance
        h = self._distance(eye_points[0], eye_points[3])
        
        # EAR calculation
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def _distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def move_mouse_precise(self, screen_pos):
        """Precise mouse movement with intelligent smoothing"""
        if not screen_pos:
            return
        
        current_mouse_pos = pyautogui.position()
        
        if self.last_mouse_pos is None:
            self.last_mouse_pos = screen_pos
            return
        
        # Calculate movement distance
        distance = math.sqrt((screen_pos[0] - current_mouse_pos[0])**2 + 
                           (screen_pos[1] - current_mouse_pos[1])**2)
        
        # Only move if distance is significant
        if distance > self.movement_threshold:
            # Adaptive smoothing based on distance
            if distance > 100:  # Large movement - less smoothing
                smoothing = self.smoothing_factor * 0.5
            else:  # Small movement - more smoothing
                smoothing = self.smoothing_factor
            
            # Apply smoothing
            smooth_x = int(self.last_mouse_pos[0] * (1 - smoothing) + screen_pos[0] * smoothing)
            smooth_y = int(self.last_mouse_pos[1] * (1 - smoothing) + screen_pos[1] * smoothing)
            
            # Move mouse
            try:
                pyautogui.moveTo(smooth_x, smooth_y, duration=0)
                self.last_mouse_pos = (smooth_x, smooth_y)
            except Exception as e:
                print(f"Mouse movement error: {e}")
    
    def save_calibration(self, filename="precise_calibration.json"):
        """Save calibration data"""
        if self.calibration_data:
            with open(filename, 'w') as f:
                json.dump(self.calibration_data, f, indent=2)
            print(f"Calibration saved to {filename}")
    
    def load_calibration(self, filename="precise_calibration.json"):
        """Load calibration data"""
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    self.calibration_data = json.load(f)
                self._build_interpolation_model(self.calibration_data)
                print(f"Calibration loaded from {filename}")
                return True
            except Exception as e:
                print(f"Error loading calibration: {e}")
        return False
    
    def _show_calibration_instructions(self, blank_img):
        """Show calibration instructions"""
        instruction_img = blank_img.copy()
        instructions = [
            "ENHANCED PRECISION CALIBRATION",
            "",
            "Instructions:",
            "1. Keep your head completely still during calibration",
            "2. Look directly at each red dot with your eyes only",
            "3. Focus on the center of each dot until it disappears",
            "4. Blink normally during calibration",
            "",
            "This will take about 2 minutes for 25 calibration points",
            "",
            "Press SPACE to start",
            "Press ESC to cancel"
        ]
        
        y_offset = self.screen_height // 2 - len(instructions) * 25
        for i, line in enumerate(instructions):
            color = (0, 255, 255) if "ENHANCED" in line else (255, 255, 255)
            font_size = 1.0 if i == 0 else 0.7
            cv2.putText(instruction_img, line,
                       (self.screen_width // 2 - 300, y_offset + i * 45),
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 2)
        
        cv2.imshow("Enhanced Calibration", instruction_img)
        
        # Wait for start
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                raise KeyboardInterrupt("Calibration cancelled")
    
    def _show_calibration_point(self, blank_img, x_pos, y_pos, point_idx, progress, status):
        """Show calibration point with progress"""
        cal_img = blank_img.copy()
        
        # Draw target point
        cv2.circle(cal_img, (x_pos, y_pos), 12, (0, 0, 255), -1)  # Red dot
        cv2.circle(cal_img, (x_pos, y_pos), 16, (255, 255, 255), 2)  # White border
        
        # Progress ring
        if progress > 0:
            end_angle = int(360 * progress)
            cv2.ellipse(cal_img, (x_pos, y_pos), (25, 25), 0, 0, end_angle, (0, 255, 0), 3)
        
        # Point info
        cv2.putText(cal_img, f"Point {point_idx + 1}/{len(self.calibration_points)}",
                   (x_pos - 80, y_pos - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(cal_img, status,
                   (x_pos - 80, y_pos + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Enhanced Calibration", cal_img)
    
    def _show_calibration_complete(self, blank_img, num_points):
        """Show calibration completion message"""
        complete_img = blank_img.copy()
        cv2.putText(complete_img, "CALIBRATION COMPLETE!",
                   (self.screen_width // 2 - 250, self.screen_height // 2 - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(complete_img, f"Successfully calibrated {num_points} points",
                   (self.screen_width // 2 - 200, self.screen_height // 2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(complete_img, "Press any key to continue...",
                   (self.screen_width // 2 - 150, self.screen_height // 2 + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        cv2.imshow("Enhanced Calibration", complete_img)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    
    def _show_calibration_failed(self, blank_img):
        """Show calibration failure message"""
        fail_img = blank_img.copy()
        cv2.putText(fail_img, "CALIBRATION FAILED",
                   (self.screen_width // 2 - 200, self.screen_height // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(fail_img, "Not enough quality calibration points collected",
                   (self.screen_width // 2 - 250, self.screen_height // 2 + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow("Enhanced Calibration", fail_img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

def main():
    """Test the enhanced calibration system"""
    print("Enhanced Eye Tracking Calibration System")
    print("=" * 50)
    
    tracker = PreciseEyeTracker()
    
    # Try to load existing calibration
    if not tracker.load_calibration():
        print("No existing calibration found. Starting new calibration...")
        try:
            calibration_result = tracker.enhanced_calibration()
            if calibration_result:
                tracker.save_calibration()
                print("✓ Enhanced calibration completed successfully!")
            else:
                print("✗ Calibration failed. Please try again.")
                return
        except KeyboardInterrupt:
            print("Calibration cancelled by user.")
            return
    else:
        print("✓ Existing calibration loaded successfully!")
    
    print("\nCalibration system ready!")
    print("You can now use this calibration with the main eye tracking system.")

if __name__ == "__main__":
    main()