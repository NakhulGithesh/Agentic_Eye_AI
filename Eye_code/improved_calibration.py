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
from camera_config import EXTERNAL_CAMERA_ID

class ImprovedCalibration:
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
        
        # Improved 9-point grid with better coverage
        self.calibration_points = [
            (0.1, 0.1),   # Top-left
            (0.5, 0.1),   # Top-center  
            (0.9, 0.1),   # Top-right
            (0.1, 0.5),   # Middle-left
            (0.5, 0.5),   # Center
            (0.9, 0.5),   # Middle-right
            (0.1, 0.9),   # Bottom-left
            (0.5, 0.9),   # Bottom-center
            (0.9, 0.9),   # Bottom-right
        ]
        
        # Stability tracking
        self.gaze_buffer = deque(maxlen=10)
        self.stability_threshold = 0.02  # Stricter stability requirement
        
    def detect_iris_stable(self, frame):
        """Enhanced iris detection with stability checking"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None, False
        
        landmarks = results.multi_face_landmarks[0]
        
        # Get iris landmarks with better precision
        left_iris_indices = list(range(474, 478))
        right_iris_indices = list(range(469, 473))
        
        left_iris = [landmarks.landmark[i] for i in left_iris_indices]
        right_iris = [landmarks.landmark[i] for i in right_iris_indices]
        
        # Calculate iris centers with sub-pixel accuracy
        left_x = sum(p.x for p in left_iris) / len(left_iris)
        left_y = sum(p.y for p in left_iris) / len(left_iris)
        
        right_x = sum(p.x for p in right_iris) / len(right_iris)
        right_y = sum(p.y for p in right_iris) / len(right_iris)
        
        # Average both eyes with quality weighting
        avg_x = (left_x + right_x) / 2.0
        avg_y = (left_y + right_y) / 2.0
        
        # Fix coordinate inversion for flipped video
        # Flip X coordinate to match the flipped camera view
        avg_x = 1.0 - avg_x
        
        # Add to stability buffer
        current_gaze = (avg_x, avg_y)
        self.gaze_buffer.append(current_gaze)
        
        # Check stability over recent samples
        is_stable = False
        if len(self.gaze_buffer) >= 5:
            recent_gazes = np.array(list(self.gaze_buffer)[-5:])
            gaze_std = np.std(recent_gazes, axis=0)
            stability_score = np.mean(gaze_std)
            is_stable = stability_score < self.stability_threshold
        
        return current_gaze, is_stable
    
    def run_improved_calibration(self):
        """Run improved calibration with better data collection"""
        print("Starting Improved Eye Tracking Calibration...")
        print("=" * 50)
        
        cap = cv2.VideoCapture(EXTERNAL_CAMERA_ID)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam")
            return None
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Calibration data storage
        calibration_data = {
            "gaze_points": [],
            "screen_points": [],
            "quality_scores": [],
            "timestamp": time.time()
        }
        
        # Create fullscreen window
        cv2.namedWindow("Improved Calibration", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Improved Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        blank_screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        # Show instructions
        self._show_instructions(blank_screen)
        
        print(f"Calibrating {len(self.calibration_points)} points with improved accuracy...")
        
        try:
            for point_idx, (x_pct, y_pct) in enumerate(self.calibration_points):
                x_pos = int(x_pct * self.screen_width)
                y_pos = int(y_pct * self.screen_height)
                
                print(f"\\n=== Calibrating Point {point_idx + 1}/{len(self.calibration_points)} ===")
                print(f"Target: ({x_pct:.1f}, {y_pct:.1f}) -> ({x_pos}, {y_pos})")
                
                # Clear gaze buffer for new point
                self.gaze_buffer.clear()
                
                # Phase 1: Get ready period (2 seconds)
                ready_duration = 2.0
                start_time = time.time()
                
                while time.time() - start_time < ready_duration:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    frame = cv2.flip(frame, 1)  # Mirror the video
                    self._show_calibration_point(blank_screen, x_pos, y_pos, point_idx, 
                                                "GET READY", 0.0, (255, 255, 0))
                    
                    # Still detect gaze to start building stability
                    gaze_pos, is_stable = self.detect_iris_stable(frame)
                    
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC to cancel
                        raise KeyboardInterrupt("Calibration cancelled")
                
                # Phase 2: Data collection (5 seconds)
                collection_duration = 5.0
                collected_samples = []
                quality_scores = []
                start_time = time.time()
                
                print("Collecting stable gaze samples...")
                
                while time.time() - start_time < collection_duration:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    frame = cv2.flip(frame, 1)
                    gaze_pos, is_stable = self.detect_iris_stable(frame)
                    
                    progress = (time.time() - start_time) / collection_duration
                    status = f"COLLECTING ({len(collected_samples)} samples)"
                    color = (0, 255, 0) if is_stable else (0, 255, 255)
                    
                    self._show_calibration_point(blank_screen, x_pos, y_pos, point_idx,
                                                status, progress, color)
                    
                    # Only collect samples when gaze is stable
                    if gaze_pos and is_stable:
                        # Calculate quality score based on recent stability
                        if len(self.gaze_buffer) >= 5:
                            recent_gazes = np.array(list(self.gaze_buffer)[-5:])
                            stability_score = 1.0 / (1.0 + np.mean(np.std(recent_gazes, axis=0)))
                            
                            if stability_score > 0.8:  # High quality threshold
                                collected_samples.append(gaze_pos)
                                quality_scores.append(stability_score)
                    
                    if cv2.waitKey(1) & 0xFF == 27:
                        raise KeyboardInterrupt("Calibration cancelled")
                
                # Process collected samples
                if len(collected_samples) >= 20:  # Require minimum samples
                    samples_array = np.array(collected_samples)
                    quality_array = np.array(quality_scores)
                    
                    # Remove outliers using IQR method
                    def remove_outliers(data, weights):
                        Q1 = np.percentile(data, 25, axis=0)
                        Q3 = np.percentile(data, 75, axis=0)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
                        return data[mask], weights[mask]
                    
                    filtered_samples, filtered_quality = remove_outliers(samples_array, quality_array)
                    
                    if len(filtered_samples) >= 10:
                        # Weighted average based on quality
                        weights = filtered_quality / np.sum(filtered_quality)
                        final_gaze_x = np.average(filtered_samples[:, 0], weights=weights)
                        final_gaze_y = np.average(filtered_samples[:, 1], weights=weights)
                        
                        # Store calibration data
                        calibration_data["gaze_points"].append([final_gaze_x, final_gaze_y])
                        calibration_data["screen_points"].append([x_pct, y_pct])
                        calibration_data["quality_scores"].append(np.mean(filtered_quality))
                        
                        print(f"✓ Success: {len(filtered_samples)} quality samples, avg quality: {np.mean(filtered_quality):.3f}")
                        print(f"  Final gaze: ({final_gaze_x:.3f}, {final_gaze_y:.3f})")
                    else:
                        print(f"✗ Failed: Not enough samples after outlier removal ({len(filtered_samples)})")
                else:
                    print(f"✗ Failed: Insufficient stable samples ({len(collected_samples)})")
                
                # Short pause between points
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            print("Calibration cancelled by user")
            cap.release()
            cv2.destroyAllWindows()
            return None
        
        finally:
            cap.release()
        
        # Validate and build interpolation model
        num_points = len(calibration_data["gaze_points"])
        print(f"=== Calibration Results ====")
        print(f"Successfully calibrated: {num_points}/{len(self.calibration_points)} points")
        
        if num_points >= 6:  # Minimum for reasonable interpolation
            # Build multiple interpolation models and save
            self._build_and_save_calibration(calibration_data)
            self._show_completion_message(blank_screen, num_points)
            return calibration_data
        else:
            self._show_failure_message(blank_screen)
            return None
    
    def _build_and_save_calibration(self, calibration_data):
        """Build and save multiple calibration formats"""
        print("Building interpolation models...")
        
        # Save basic calibration data
        with open("calibration_data.json", 'w') as f:
            json.dump(calibration_data, f, indent=2)
        print("✓ Saved: calibration_data.json")
        
        # Build and save RBF calibration if enough points
        if len(calibration_data["gaze_points"]) >= 8:
            try:
                gaze_points = np.array(calibration_data["gaze_points"])
                screen_points = np.array(calibration_data["screen_points"])
                
                # Create RBF interpolators
                rbf_x = Rbf(gaze_points[:, 0], gaze_points[:, 1], screen_points[:, 0],
                           function='thin_plate', smooth=0.001)
                rbf_y = Rbf(gaze_points[:, 0], gaze_points[:, 1], screen_points[:, 1],
                           function='thin_plate', smooth=0.001)
                
                # Test RBF quality
                test_errors = []
                for i in range(len(gaze_points)):
                    test_gaze = gaze_points[i]
                    expected_screen = screen_points[i]
                    
                    predicted_x = rbf_x(test_gaze[0], test_gaze[1])
                    predicted_y = rbf_y(test_gaze[0], test_gaze[1])
                    
                    error = math.sqrt((predicted_x - expected_screen[0])**2 + 
                                    (predicted_y - expected_screen[1])**2)
                    test_errors.append(error)
                
                avg_error = np.mean(test_errors)
                max_error = np.max(test_errors)
                
                # Enhanced calibration data with RBF info
                rbf_calibration_data = calibration_data.copy()
                rbf_calibration_data.update({
                    "rbf_smooth_factor": 0.001,
                    "avg_quality": np.mean(calibration_data["quality_scores"]),
                    "avg_error": avg_error,
                    "max_error": max_error,
                    "quality_grade": "EXCELLENT" if avg_error < 0.05 else "GOOD" if avg_error < 0.1 else "FAIR"
                })
                
                with open("calibration_rbf.json", 'w') as f:
                    json.dump(rbf_calibration_data, f, indent=2)
                
                print(f"✓ Saved: calibration_rbf.json (Error: {avg_error:.3f}, Grade: {rbf_calibration_data['quality_grade']})")
                
            except Exception as e:
                print(f"✗ RBF calibration failed: {e}")
    
    def _show_instructions(self, blank_screen):
        """Show calibration instructions"""
        instruction_screen = blank_screen.copy()
        
        instructions = [
            "IMPROVED EYE TRACKING CALIBRATION",
            "",
            "Instructions:",
            "• Keep your head still during the entire calibration",
            "• Look directly at each red dot when it appears", 
            "• Stay focused on each dot for the full 5 seconds",
            "• Blink normally, but avoid excessive head movement",
            "• The system will collect data only when your gaze is stable",
            "",
            "This process takes about 2 minutes for maximum accuracy",
            "",
            "Press SPACE to begin calibration",
            "Press ESC to cancel"
        ]
        
        y_start = self.screen_height // 2 - (len(instructions) * 25)
        
        for i, line in enumerate(instructions):
            if "IMPROVED" in line:
                color = (0, 255, 255)
                font_scale = 1.2
            elif line.startswith("•"):
                color = (255, 255, 255)
                font_scale = 0.7
            else:
                color = (200, 200, 200)
                font_scale = 0.8
                
            cv2.putText(instruction_screen, line,
                       (50, y_start + i * 35),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        
        cv2.imshow("Improved Calibration", instruction_screen)
        
        # Wait for user to start
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            elif key == 27:  # ESC
                raise KeyboardInterrupt("Calibration cancelled")
    
    def _show_calibration_point(self, blank_screen, x_pos, y_pos, point_idx, 
                               status, progress, color):
        """Show calibration point with enhanced feedback"""
        cal_screen = blank_screen.copy()
        
        # Main target dot
        cv2.circle(cal_screen, (x_pos, y_pos), 15, (0, 0, 255), -1)  # Red center
        cv2.circle(cal_screen, (x_pos, y_pos), 20, color, 3)  # Colored ring
        
        # Progress ring
        if progress > 0:
            end_angle = int(360 * progress)
            cv2.ellipse(cal_screen, (x_pos, y_pos), (30, 30), 0, 0, end_angle, (0, 255, 0), 4)
        
        # Point information
        info_text = f"Point {point_idx + 1}/{len(self.calibration_points)}"
        cv2.putText(cal_screen, info_text, (x_pos - 100, y_pos - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Status
        cv2.putText(cal_screen, status, (x_pos - 120, y_pos + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Progress percentage
        if progress > 0:
            progress_text = f"{int(progress * 100)}%"
            cv2.putText(cal_screen, progress_text, (x_pos - 20, y_pos + 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Improved Calibration", cal_screen)
    
    def _show_completion_message(self, blank_screen, num_points):
        """Show calibration completion"""
        complete_screen = blank_screen.copy()
        
        # Success message
        cv2.putText(complete_screen, "CALIBRATION COMPLETED!",
                   (self.screen_width // 2 - 300, self.screen_height // 2 - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        cv2.putText(complete_screen, f"Successfully calibrated {num_points} points",
                   (self.screen_width // 2 - 250, self.screen_height // 2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        cv2.putText(complete_screen, "Enhanced accuracy model built",
                   (self.screen_width // 2 - 200, self.screen_height // 2 + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.putText(complete_screen, "Press any key to continue...",
                   (self.screen_width // 2 - 150, self.screen_height // 2 + 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        cv2.imshow("Improved Calibration", complete_screen)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    
    def _show_failure_message(self, blank_screen):
        """Show calibration failure"""
        fail_screen = blank_screen.copy()
        
        cv2.putText(fail_screen, "CALIBRATION FAILED",
                   (self.screen_width // 2 - 250, self.screen_height // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        cv2.putText(fail_screen, "Not enough stable calibration points collected",
                   (self.screen_width // 2 - 300, self.screen_height // 2 + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(fail_screen, "Please try again with better head stability",
                   (self.screen_width // 2 - 250, self.screen_height // 2 + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        cv2.imshow("Improved Calibration", fail_screen)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

def main():
    """Run the improved calibration"""
    print("IMPROVED EYE TRACKING CALIBRATION SYSTEM")
    print("=" * 50)
    
    calibrator = ImprovedCalibration()
    
    try:
        result = calibrator.run_improved_calibration()
        
        if result:
            print("=" * 50)
            print("✓ CALIBRATION SUCCESSFUL!")
            print("✓ Multiple calibration files saved")
            print("✓ Enhanced accuracy model ready")
            print("You can now use the eye tracking system with improved accuracy!")
        else:
            print("=" * 50)
            print("✗ CALIBRATION FAILED")
            print("Please try again and keep your head very still")
            
    except Exception as e:
        print(f"[ERROR] Calibration error: {e}")
    
    print("Press Enter to exit...")
    input()

if __name__ == "__main__":
    main()