from camera_config import EXTERNAL_CAMERA_ID
import cv2
import mediapipe as mp
import numpy as np
import json
import os
import pyautogui
from scipy.interpolate import Rbf
import time
import math
from collections import deque

class RelativeEyeTracker:
    def __init__(self):
        # MediaPipe initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Screen properties
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False
        
        # Enhanced calibration points
        self.calibration_points = self._generate_calibration_grid()
        
        # Tracking buffers for stability
        self.gaze_buffer = deque(maxlen=8)
        self.stable_gaze_buffer = deque(maxlen=3)
        
        # Calibration data
        self.calibration_data = None
        self.rbf_interpolator_x = None
        self.rbf_interpolator_y = None
        
        # Eye landmark indices for relative positioning
        self.left_eye_landmarks = {
            'outer_corner': 33,
            'inner_corner': 133,
            'top': 159,
            'bottom': 145,
            'iris': [474, 475, 476, 477]
        }
        
        self.right_eye_landmarks = {
            'outer_corner': 362,
            'inner_corner': 263,
            'top': 386,
            'bottom': 374,
            'iris': [469, 470, 471, 472]
        }
        
    def _generate_calibration_grid(self):
        """Generate a comprehensive 25-point calibration grid"""
        points = []
        for y in [0.05, 0.25, 0.5, 0.75, 0.95]:
            for x in [0.05, 0.25, 0.5, 0.75, 0.95]:
                points.append((x, y))
        return points
    
    def calculate_relative_iris_position(self, landmarks):
        """Calculate iris position relative to eye socket landmarks - same as controller"""
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
        
        # Calculate relative positions within eye sockets
        if left_eye_width > 0:
            left_relative_x = (left_iris_x - left_outer.x) / left_eye_width
        else:
            left_relative_x = 0.5
            
        if left_eye_height > 0:
            left_relative_y = (left_iris_y - left_top.y) / left_eye_height
        else:
            left_relative_y = 0.5
        
        if right_eye_width > 0:
            right_relative_x = (right_iris_x - right_outer.x) / right_eye_width
        else:
            right_relative_x = 0.5
            
        if right_eye_height > 0:
            right_relative_y = (right_iris_y - right_top.y) / right_eye_height
        else:
            right_relative_y = 0.5
        
        # Average both eyes
        avg_relative_x = (left_relative_x + right_relative_x) / 2.0
        avg_relative_y = (left_relative_y + right_relative_y) / 2.0
        
        # Use direct X coordinate for natural mapping
        flipped_x = avg_relative_x
        
        # Clamp values
        flipped_x = max(0.0, min(1.0, flipped_x))
        avg_relative_y = max(0.0, min(1.0, avg_relative_y))
        
        return (flipped_x, avg_relative_y)
    
    def detect_iris_relative(self, frame):
        """Enhanced relative iris detection with stability"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None, None
        
        landmarks = results.multi_face_landmarks[0]
        
        # Calculate relative iris position
        relative_gaze = self.calculate_relative_iris_position(landmarks)
        
        if not relative_gaze:
            return None, landmarks
        
        # Add to buffer for temporal stability
        current_gaze = relative_gaze
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
                if distance < 0.03:  # Adjusted for relative coordinates
                    stable_x = last_stable[0] * 0.7 + stable_x * 0.3
                    stable_y = last_stable[1] * 0.7 + stable_y * 0.3
            
            stable_gaze = (stable_x, stable_y)
            self.stable_gaze_buffer.append(stable_gaze)
            return stable_gaze, landmarks
        
        return current_gaze, landmarks
    
    def enhanced_calibration(self):
        """Enhanced calibration with relative positioning"""
        print("Starting Enhanced Relative Calibration...")
        print("Using eye socket-relative positioning for better accuracy")
        
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
            "timestamp": time.time(),
            "tracking_mode": "relative_positioning"
        }
        
        # Create fullscreen calibration window
        cv2.namedWindow("Relative Calibration", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Relative Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        blank_img = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        # Show instructions
        self._show_calibration_instructions(blank_img)
        
        print(f"Calibrating {len(self.calibration_points)} points with relative positioning...")
        
        for point_idx, (x_pct, y_pct) in enumerate(self.calibration_points):
            x_pos = int(x_pct * self.screen_width)
            y_pos = int(y_pct * self.screen_height)
            
            print(f"Calibrating point {point_idx + 1}/{len(self.calibration_points)}")
            
            # Collect high-quality samples
            gaze_samples = []
            quality_scores = []
            calibration_duration = 4.0
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
                relative_gaze, landmarks = self.detect_iris_relative(frame)
                
                if relative_gaze and landmarks:
                    # Quality check based on face detection confidence
                    quality_score = self._calculate_gaze_quality(landmarks, relative_gaze)
                    if quality_score > 0.7:  # Only accept high-quality samples
                        gaze_samples.append(relative_gaze)
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
            if len(gaze_samples) >= 20:
                gaze_array = np.array(gaze_samples)
                quality_array = np.array(quality_scores)
                
                # Remove outliers
                mean_gaze = np.mean(gaze_array, axis=0)
                distances = np.sqrt(np.sum((gaze_array - mean_gaze)**2, axis=1))
                distance_threshold = np.percentile(distances, 75)
                
                valid_indices = distances <= distance_threshold
                filtered_gaze = gaze_array[valid_indices]
                filtered_quality = quality_array[valid_indices]
                
                if len(filtered_gaze) >= 10:
                    # Weighted average based on quality scores
                    weights = filtered_quality / np.sum(filtered_quality)
                    final_gaze_x = np.average(filtered_gaze[:, 0], weights=weights)
                    final_gaze_y = np.average(filtered_gaze[:, 1], weights=weights)
                    
                    calibration_data["screen_points"].append([x_pct, y_pct])
                    calibration_data["gaze_points"].append([final_gaze_x, final_gaze_y])
                    calibration_data["quality_scores"].append(np.mean(filtered_quality))
                    
                    print(f"  + Point {point_idx + 1}: {len(filtered_gaze)} quality samples (relative)")
                else:
                    print(f"  - Point {point_idx + 1}: Not enough quality data after filtering")
            else:
                print(f"  - Point {point_idx + 1}: Insufficient samples collected")
        
        cap.release()
        
        # Build enhanced interpolation model
        if len(calibration_data["gaze_points"]) >= 15:
            self._build_interpolation_model(calibration_data)
            self._show_calibration_complete(blank_img, len(calibration_data["gaze_points"]))
            self.calibration_data = calibration_data
            return calibration_data
        else:
            self._show_calibration_failed(blank_img)
            return None
    
    def _calculate_gaze_quality(self, landmarks, relative_gaze):
        """Calculate quality score for relative gaze sample"""
        # Check face stability
        face_landmarks = [landmarks.landmark[i] for i in [1, 33, 61, 291, 199]]
        face_points = [(l.x, l.y) for l in face_landmarks]
        
        face_array = np.array(face_points)
        face_std = np.std(face_array, axis=0)
        stability_score = max(0, 1.0 - np.mean(face_std) * 10)
        
        # Check gaze consistency with recent samples
        consistency_score = 1.0
        if len(self.gaze_buffer) > 3:
            recent_gazes = np.array(list(self.gaze_buffer)[-3:])
            gaze_std = np.std(recent_gazes, axis=0)
            consistency_score = max(0, 1.0 - np.mean(gaze_std) * 15)  # Adjusted for relative coords
        
        return (stability_score + consistency_score) / 2.0
    
    def _build_interpolation_model(self, calibration_data):
        """Build RBF interpolation model for relative positioning"""
        gaze_points = np.array(calibration_data["gaze_points"])
        screen_points = np.array(calibration_data["screen_points"])
        
        self.rbf_interpolator_x = Rbf(
            gaze_points[:, 0], gaze_points[:, 1], screen_points[:, 0],
            function='thin_plate', smooth=0.001
        )
        self.rbf_interpolator_y = Rbf(
            gaze_points[:, 0], gaze_points[:, 1], screen_points[:, 1],
            function='thin_plate', smooth=0.001
        )
        
        print("Relative positioning interpolation model built successfully!")
    
    def save_calibration(self, filename="precise_calibration.json"):
        """Save relative calibration data"""
        if self.calibration_data:
            with open(filename, 'w') as f:
                json.dump(self.calibration_data, f, indent=2)
            print(f"Relative calibration saved to {filename}")
    
    def _show_calibration_instructions(self, blank_img):
        """Show calibration instructions"""
        instruction_img = blank_img.copy()
        instructions = [
            "RELATIVE POSITIONING CALIBRATION",
            "",
            "Instructions:",
            "1. Keep your head completely still during calibration",
            "2. Look directly at each red dot with your eyes only",
            "3. Focus on the center of each dot until it disappears",
            "4. This system uses eye socket-relative positioning",
            "5. Works better across different head positions",
            "",
            "This will take about 2 minutes for 25 calibration points",
            "",
            "Press SPACE to start",
            "Press ESC to cancel"
        ]
        
        y_offset = self.screen_height // 2 - len(instructions) * 25
        for i, line in enumerate(instructions):
            color = (0, 255, 255) if "RELATIVE" in line else (255, 255, 255)
            font_size = 1.0 if i == 0 else 0.7
            cv2.putText(instruction_img, line,
                       (self.screen_width // 2 - 350, y_offset + i * 45),
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 2)
        
        cv2.imshow("Relative Calibration", instruction_img)
        
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
        cv2.putText(cal_img, f"Point {point_idx + 1}/{len(self.calibration_points)} (Relative)",
                   (x_pos - 120, y_pos - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(cal_img, status,
                   (x_pos - 80, y_pos + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Relative Calibration", cal_img)
    
    def _show_calibration_complete(self, blank_img, num_points):
        """Show calibration completion message"""
        complete_img = blank_img.copy()
        cv2.putText(complete_img, "RELATIVE CALIBRATION COMPLETE!",
                   (self.screen_width // 2 - 300, self.screen_height // 2 - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(complete_img, f"Successfully calibrated {num_points} points with relative positioning",
                   (self.screen_width // 2 - 350, self.screen_height // 2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(complete_img, "Press any key to continue...",
                   (self.screen_width // 2 - 150, self.screen_height // 2 + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        cv2.imshow("Relative Calibration", complete_img)
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
        
        cv2.imshow("Relative Calibration", fail_img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

def main():
    """Test the relative calibration system"""
    print("Relative Eye Tracking Calibration System")
    print("=" * 50)
    print("Features:")
    print("+ Eye socket-relative positioning")
    print("+ Flipped X-coordinate for natural mapping")
    print("+ Person-independent tracking")
    print("=" * 50)
    
    tracker = RelativeEyeTracker()
    
    print("Starting relative positioning calibration...")
    try:
        calibration_result = tracker.enhanced_calibration()
        if calibration_result:
            tracker.save_calibration()
            print("+ Relative calibration completed successfully!")
        else:
            print("- Calibration failed. Please try again.")
            return
    except KeyboardInterrupt:
        print("Calibration cancelled by user.")
        return
    
    print("\nRelative calibration system ready!")
    print("You can now use this calibration with the relative eye tracking system.")

if __name__ == "__main__":
    main()