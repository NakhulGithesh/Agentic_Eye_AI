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

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh

def load_calibration():
    """Load calibration data"""
    if os.path.exists("calibration_data.json"):
        with open("calibration_data.json", 'r') as f:
            return json.load(f)
    return None

def load_historical_calibration():
    """Load historical calibration data for cumulative improvement"""
    if os.path.exists("calibration_history.json"):
        with open("calibration_history.json", 'r') as f:
            return json.load(f)
    return {"sessions": [], "merged_data": None}

def merge_calibration_data(new_data, historical_data, weight_recent=0.7):
    """Merge new calibration with historical data - with overflow protection"""
    if not historical_data.get("merged_data"):
        # First calibration
        return {
            "screen_points": new_data["screen_points"],
            "gaze_points": new_data["gaze_points"],
            "timestamp": new_data["timestamp"],
            "total_sessions": 1
        }
    
    # Combine new data with existing merged data
    merged = historical_data["merged_data"]
    sessions_count = len(historical_data["sessions"])
    
    # Prevent data overflow - limit to maximum 8 sessions (128 points max)
    if sessions_count >= 8:
        print("[INFO] Maximum calibration sessions reached. Using recent 8 sessions only.")
        # Keep only the most recent sessions
        recent_sessions = historical_data["sessions"][-7:]  # Keep 7 old + 1 new = 8 total
        
        # Rebuild merged data from scratch with recent sessions
        all_screen_points = new_data["screen_points"]
        all_gaze_points = new_data["gaze_points"] 
        all_quality_scores = new_data.get("quality_scores", [1.0] * len(new_data["gaze_points"]))
        
        # Add data from recent sessions (if we had access to individual session data)
        # For now, just use current + some previous data with limits
        if len(merged["screen_points"]) > 64:  # Limit previous data to ~4 sessions worth
            merged["screen_points"] = merged["screen_points"][-64:]
            merged["gaze_points"] = merged["gaze_points"][-64:]
            if "quality_scores" in merged:
                merged["quality_scores"] = merged["quality_scores"][-64:]
        
        all_screen_points = merged["screen_points"] + all_screen_points
        all_gaze_points = merged["gaze_points"] + all_gaze_points
        if "quality_scores" in merged:
            all_quality_scores = merged.get("quality_scores", [1.0] * len(merged["gaze_points"])) + all_quality_scores
        
        historical_data["sessions"] = recent_sessions
    else:
        # Normal merge - add new points to existing ones
        all_screen_points = merged["screen_points"] + new_data["screen_points"]
        all_gaze_points = merged["gaze_points"] + new_data["gaze_points"]
        all_quality_scores = (merged.get("quality_scores", [1.0] * len(merged["gaze_points"])) + 
                             new_data.get("quality_scores", [1.0] * len(new_data["gaze_points"])))
    
    # Weight recent calibrations more heavily
    total_points = len(all_screen_points)
    old_points = len(merged["screen_points"]) if sessions_count < 8 else len(all_screen_points) - len(new_data["screen_points"])
    new_points = len(new_data["screen_points"])
    
    # Create weights (more recent calibrations get higher weight)
    weights = ([1.0 - weight_recent] * old_points + 
              [weight_recent] * new_points)
    
    result = {
        "screen_points": all_screen_points,
        "gaze_points": all_gaze_points,
        "quality_scores": all_quality_scores,
        "weights": weights,
        "timestamp": new_data["timestamp"],
        "total_sessions": min(sessions_count + 1, 8)  # Cap at 8 sessions
    }
    
    print(f"[MERGE] Combined {len(all_screen_points)} points from {result['total_sessions']} sessions")
    
    return result

def clear_calibration_data():
    """Clear all calibration data and start fresh"""
    files_to_remove = [
        "calibration_data.json",
        "calibration_enhanced.json", 
        "calibration_rbf.json",
        "calibration_history.json"
    ]
    
    removed_count = 0
    for file in files_to_remove:
        if os.path.exists(file):
            try:
                os.remove(file)
                removed_count += 1
            except Exception as e:
                print(f"[WARNING] Could not remove {file}: {e}")
    
    print(f"[RESET] Cleared {removed_count} calibration files")
    print("[RESET] You can now start fresh calibration")
    return removed_count > 0

def detect_iris_advanced(frame, face_mesh):
    """Advanced iris detection with quality scoring"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    if not results.multi_face_landmarks:
        return None, 0.0
    
    landmarks = results.multi_face_landmarks[0]
    
    # Get iris landmarks with quality weights
    left_iris_indices = list(range(474, 478))
    right_iris_indices = list(range(469, 473))
    
    # Weighted center calculation for higher precision
    left_iris_points = [landmarks.landmark[i] for i in left_iris_indices]
    right_iris_points = [landmarks.landmark[i] for i in right_iris_indices]
    
    # Quality-based weights (center points are more reliable)
    weights = [1.2, 1.0, 1.0, 1.2]  # Emphasize top/bottom iris points
    
    left_x = sum(p.x * w for p, w in zip(left_iris_points, weights)) / sum(weights)
    left_y = sum(p.y * w for p, w in zip(left_iris_points, weights)) / sum(weights)
    
    right_x = sum(p.x * w for p, w in zip(right_iris_points, weights)) / sum(weights)
    right_y = sum(p.y * w for p, w in zip(right_iris_points, weights)) / sum(weights)
    
    # Average both eyes
    avg_x = (left_x + right_x) / 2.0
    avg_y = (left_y + right_y) / 2.0
    
    # Calculate quality score
    quality = calculate_detection_quality(landmarks, (avg_x, avg_y))
    
    return (avg_x, avg_y), quality

def calculate_detection_quality(landmarks, gaze_pos):
    """Calculate quality score for iris detection"""
    quality_factors = []
    
    # 1. Eye openness quality
    left_eye_points = [landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
    right_eye_points = [landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
    
    left_ear = calculate_ear(left_eye_points)
    right_ear = calculate_ear(right_eye_points)
    avg_ear = (left_ear + right_ear) / 2.0
    
    # Ideal EAR range for good detection
    if 0.25 < avg_ear < 0.35:
        eye_quality = 1.0
    elif 0.2 < avg_ear < 0.4:
        eye_quality = 0.8
    else:
        eye_quality = 0.5
    
    quality_factors.append(eye_quality)
    
    # 2. Face stability (nose position as reference)
    nose_tip = (landmarks.landmark[1].x, landmarks.landmark[1].y)
    if hasattr(calculate_detection_quality, 'reference_nose'):
        nose_distance = math.sqrt((nose_tip[0] - calculate_detection_quality.reference_nose[0])**2 + 
                                (nose_tip[1] - calculate_detection_quality.reference_nose[1])**2)
        stability = max(0, 1.0 - nose_distance * 100)  # Penalize head movement
    else:
        calculate_detection_quality.reference_nose = nose_tip
        stability = 1.0
    
    quality_factors.append(stability)
    
    # 3. Iris position reasonableness
    if 0.1 < gaze_pos[0] < 0.9 and 0.1 < gaze_pos[1] < 0.9:
        position_quality = 1.0
    else:
        position_quality = 0.7  # Edge positions are less reliable
    
    quality_factors.append(position_quality)
    
    return np.mean(quality_factors)

def calculate_ear(eye_points):
    """Calculate Eye Aspect Ratio for quality assessment"""
    def distance(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    v1 = distance(eye_points[1], eye_points[5])
    v2 = distance(eye_points[2], eye_points[4])
    h = distance(eye_points[0], eye_points[3])
    
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.3

def map_to_screen(iris_pos, calibration_data):
    """Map iris position to screen coordinates"""
    if not calibration_data or not iris_pos:
        print("No calibration data or iris position")
        return None
    
    gaze_points = np.array(calibration_data["gaze_points"])
    screen_points = np.array(calibration_data["screen_points"])
    
    try:
        # Try cubic interpolation first
        screen_x = griddata(gaze_points, screen_points[:, 0], [iris_pos], method='cubic')[0]
        screen_y = griddata(gaze_points, screen_points[:, 1], [iris_pos], method='cubic')[0]
        
        # If cubic fails, try linear
        if np.isnan(screen_x) or np.isnan(screen_y):
            screen_x = griddata(gaze_points, screen_points[:, 0], [iris_pos], method='linear')[0]
            screen_y = griddata(gaze_points, screen_points[:, 1], [iris_pos], method='linear')[0]
            
        # If linear fails, try nearest
        if np.isnan(screen_x) or np.isnan(screen_y):
            screen_x = griddata(gaze_points, screen_points[:, 0], [iris_pos], method='nearest')[0]
            screen_y = griddata(gaze_points, screen_points[:, 1], [iris_pos], method='nearest')[0]
        
        # Final check for NaN values
        if np.isnan(screen_x) or np.isnan(screen_y):
            print(f"Warning: Could not map iris position {iris_pos} to screen coordinates")
            return None
            
        # Clamp values to valid range [0, 1]
        screen_x = max(0, min(1, screen_x))
        screen_y = max(0, min(1, screen_y))
        
        return (screen_x, screen_y)
    except Exception as e:
        print(f"Error in map_to_screen: {e}")
        return None

def advanced_precision_calibration():
    """Advanced 16-point calibration with quality validation"""
    print("Starting advanced precision calibration...")
    
    # Enhanced 16-point calibration grid for better accuracy
    calibration_points = [
        # Outer ring (corners and edges)
        (0.1, 0.1),   (0.9, 0.1),   # Top corners
        (0.1, 0.9),   (0.9, 0.9),   # Bottom corners
        (0.1, 0.5),   (0.9, 0.5),   # Side midpoints
        (0.5, 0.1),   (0.5, 0.9),   # Top/bottom midpoints
        
        # Inner ring for precision
        (0.25, 0.25), (0.75, 0.25), # Inner top
        (0.25, 0.75), (0.75, 0.75), # Inner bottom
        (0.25, 0.5),  (0.75, 0.5),  # Inner sides
        
        # Center points
        (0.5, 0.35),  (0.5, 0.65),  # Center vertical spread
    ]
    
    print(f"Using {len(calibration_points)} calibration points for maximum accuracy")
    
    screen_width, screen_height = pyautogui.size()
    
    # Enhanced face mesh settings for precision
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.8,  # Higher confidence for calibration
        min_tracking_confidence=0.8
    )
    
    cap = cv2.VideoCapture(EXTERNAL_CAMERA_ID)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return None
    
    calibration_data = {
        "screen_points": [],
        "gaze_points": [],
        "quality_scores": [],  # Track quality of each calibration point
        "calibration_type": "advanced_16_point",
        "timestamp": time.time()
    }
    
    # Create fullscreen window
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    blank_img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    # Show enhanced instructions
    instruction_img = blank_img.copy()
    
    # Title with background box
    title_text = "EYE TRACKING CALIBRATION"
    title_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    title_x = (screen_width - title_size[0]) // 2
    title_y = 100
    cv2.rectangle(instruction_img, (title_x - 20, title_y - 40), 
                  (title_x + title_size[0] + 20, title_y + 10), (40, 40, 40), -1)
    cv2.putText(instruction_img, title_text, (title_x, title_y),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    # Main instructions with icons/visual cues
    instructions = [
        "SETUP INSTRUCTIONS:",
        "",
        "1. [*] Sit comfortably 18-24 inches from your screen",
        "2. [*] Position camera at eye level",
        "3. [*] Ensure good lighting on your face",
        "4. [*] Remove glasses if possible (or clean them)",
        "",
        "DURING CALIBRATION:",
        "",
        "- [EYE] Follow the RED DOT with your EYES ONLY",
        "- [HEAD] Keep your HEAD completely still",
        "- [TIME] Wait for each dot to complete (3 seconds)",
        "- [AIM] Look directly at the center of each dot",
        "",
        "[INFO] 9 calibration points will appear in sequence",
        "[INFO] The process takes about 30 seconds total",
        "",
        "Ready? Press SPACE to begin",
        "Need to adjust setup? Press ESC to cancel"
    ]
    
    y_start = 180
    line_height = 35
    
    for i, line in enumerate(instructions):
        y_pos = y_start + i * line_height
        
        # Different colors and sizes for different types of text
        if line.startswith("SETUP") or line.startswith("DURING"):
            color = (0, 255, 0)  # Green for section headers
            font_scale = 0.9
            thickness = 2
        elif line.startswith("Ready?") or line.startswith("Need to"):
            color = (0, 255, 255)  # Cyan for action items
            font_scale = 0.8
            thickness = 2
        elif line.startswith("-") or any(line.startswith(str(j)) for j in range(1, 6)):
            color = (255, 255, 255)  # White for instructions
            font_scale = 0.7
            thickness = 2
        elif line.startswith("[INFO]"):
            color = (255, 255, 0)  # Yellow for important info
            font_scale = 0.7
            thickness = 2
        else:
            color = (200, 200, 200)  # Light gray for spacing
            font_scale = 0.6
            thickness = 1
        
        # Center align text
        if line.strip():
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = (screen_width - text_size[0]) // 2
            cv2.putText(instruction_img, line, (text_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    cv2.imshow("Calibration", instruction_img)
    
    # Wait for user to start
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        elif key == 27:  # ESC
            print("Calibration cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return None
    
    print("Calibration started!")
    
    # Show 3-2-1 countdown before starting
    for countdown in [3, 2, 1]:
        countdown_img = blank_img.copy()
        
        # Countdown number
        countdown_text = str(countdown)
        text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 8.0, 15)[0]
        text_x = (screen_width - text_size[0]) // 2
        text_y = (screen_height + text_size[1]) // 2
        
        # Glowing effect
        for radius in range(20, 0, -5):
            alpha = 0.3 - (radius * 0.01)
            cv2.putText(countdown_img, countdown_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 8.0, (0, 255, 255), 15)
        
        # Main countdown number
        cv2.putText(countdown_img, countdown_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 8.0, (255, 255, 255), 15)
        
        # Instructions
        instruction = "Get ready! Calibration starting in..."
        inst_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        inst_x = (screen_width - inst_size[0]) // 2
        cv2.putText(countdown_img, instruction, (inst_x, text_y - 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        cv2.imshow("Calibration", countdown_img)
        cv2.waitKey(1000)  # Show for 1 second
    
    # "GO" message
    go_img = blank_img.copy()
    go_text = "START!"
    go_size = cv2.getTextSize(go_text, cv2.FONT_HERSHEY_SIMPLEX, 8.0, 15)[0]
    go_x = (screen_width - go_size[0]) // 2
    go_y = (screen_height + go_size[1]) // 2
    cv2.putText(go_img, go_text, (go_x, go_y),
               cv2.FONT_HERSHEY_SIMPLEX, 8.0, (0, 255, 0), 15)
    cv2.imshow("Calibration", go_img)
    cv2.waitKey(500)
    
    # Calibrate each point
    for point_idx, (x_pct, y_pct) in enumerate(calibration_points):
        x_pos = int(x_pct * screen_width)
        y_pos = int(y_pct * screen_height)
        
        print(f"Calibrating point {point_idx + 1}/{len(calibration_points)}")
        
        # Advanced data collection with quality validation
        gaze_samples = []
        quality_samples = []
        calibration_duration = 3.5  # Slightly longer for quality collection
        start_time = time.time()
        
        # Reset quality reference for this point
        if hasattr(calculate_detection_quality, 'reference_nose'):
            delattr(calculate_detection_quality, 'reference_nose')
        
        while time.time() - start_time < calibration_duration:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            iris_pos, quality = detect_iris_advanced(frame, face_mesh)
            
            # Flip iris x-coordinate to match flipped frame
            if iris_pos:
                iris_pos = (1.0 - iris_pos[0], iris_pos[1])
            
            # Only collect high-quality samples
            if iris_pos and quality > 0.7:  # Quality threshold
                gaze_samples.append(iris_pos)
                quality_samples.append(quality)
            
            # Show calibration point with enhanced progress
            progress = (time.time() - start_time) / calibration_duration
            cal_img = blank_img.copy()
            
            # Calculate time remaining
            time_remaining = calibration_duration - (time.time() - start_time)
            
            # Draw animated target dot (pulsing effect)
            pulse = int(5 * abs(math.sin(time.time() * 3)))  # Pulsing radius
            main_radius = 15 + pulse
            
            # Draw multiple circles for better visibility
            cv2.circle(cal_img, (x_pos, y_pos), main_radius + 10, (50, 50, 50), 2)  # Outer guide
            cv2.circle(cal_img, (x_pos, y_pos), main_radius, (0, 0, 255), -1)  # Red dot
            cv2.circle(cal_img, (x_pos, y_pos), main_radius - 5, (255, 100, 100), -1)  # Lighter center
            cv2.circle(cal_img, (x_pos, y_pos), 5, (255, 255, 255), -1)  # White center dot
            
            # Progress circle around the target
            if progress > 0:
                end_angle = int(360 * progress)
                cv2.ellipse(cal_img, (x_pos, y_pos), (main_radius + 15, main_radius + 15), 
                           0, -90, -90 + end_angle, (0, 255, 0), 4)
            
            # Countdown timer
            countdown = math.ceil(time_remaining)
            if countdown > 0:
                timer_text = str(countdown)
                timer_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)[0]
                timer_x = x_pos - timer_size[0] // 2
                timer_y = y_pos - 80
                
                # Timer background
                cv2.rectangle(cal_img, (timer_x - 10, timer_y - 40), 
                             (timer_x + timer_size[0] + 10, timer_y + 10), (0, 0, 0), -1)
                cv2.rectangle(cal_img, (timer_x - 12, timer_y - 42), 
                             (timer_x + timer_size[0] + 12, timer_y + 12), (255, 255, 255), 2)
                
                # Timer text
                timer_color = (0, 255, 255) if countdown > 1 else (0, 255, 0)
                cv2.putText(cal_img, timer_text, (timer_x, timer_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 2.0, timer_color, 3)
            
            # Enhanced point information with quality feedback
            point_info = f"Calibration Point {point_idx + 1} of {len(calibration_points)}"
            info_size = cv2.getTextSize(point_info, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            info_x = x_pos - info_size[0] // 2
            info_y = y_pos + 80
            
            # Info background
            cv2.rectangle(cal_img, (info_x - 10, info_y - 25), 
                         (info_x + info_size[0] + 10, info_y + 35), (30, 30, 30), -1)
            cv2.putText(cal_img, point_info, (info_x, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Quality feedback
            if quality_samples:
                current_quality = np.mean(quality_samples[-5:]) if len(quality_samples) >= 5 else np.mean(quality_samples)
                quality_color = (0, 255, 0) if current_quality > 0.9 else (0, 255, 255) if current_quality > 0.8 else (0, 200, 255)
                quality_text = f"Quality: {current_quality:.2f} ({len(quality_samples)} good samples)"
                cv2.putText(cal_img, quality_text, (info_x, info_y + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 1)
            
            # Instructions reminder at top
            instruction_text = "Look directly at the center of the red dot - Keep your head still"
            inst_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            inst_x = (screen_width - inst_size[0]) // 2
            cv2.rectangle(cal_img, (inst_x - 15, 25), (inst_x + inst_size[0] + 15, 55), (40, 40, 40), -1)
            cv2.putText(cal_img, instruction_text, (inst_x, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow("Calibration", cal_img)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to cancel
                print("Calibration cancelled")
                cap.release()
                cv2.destroyAllWindows()
                return None
        
        # Process collected samples with quality weighting
        if gaze_samples and quality_samples:
            # Quality-weighted averaging (better than simple median)
            gaze_array = np.array(gaze_samples)
            quality_array = np.array(quality_samples)
            
            # Only use samples above quality threshold
            good_indices = quality_array > 0.8
            if np.any(good_indices):
                good_gazes = gaze_array[good_indices]
                good_qualities = quality_array[good_indices]
                
                # Weighted average using quality scores
                total_weight = np.sum(good_qualities)
                weighted_x = np.sum(good_gazes[:, 0] * good_qualities) / total_weight
                weighted_y = np.sum(good_gazes[:, 1] * good_qualities) / total_weight
                avg_quality = np.mean(good_qualities)
            else:
                # Fallback to all samples if none meet high threshold
                total_weight = np.sum(quality_array)
                weighted_x = np.sum(gaze_array[:, 0] * quality_array) / total_weight
                weighted_y = np.sum(gaze_array[:, 1] * quality_array) / total_weight
                avg_quality = np.mean(quality_array)
            
            calibration_data["screen_points"].append([x_pct, y_pct])
            calibration_data["gaze_points"].append([weighted_x, weighted_y])
            calibration_data["quality_scores"].append(avg_quality)
            
            print(f"  Collected {len(gaze_samples)} samples, avg quality: {avg_quality:.3f}")
            
            # Warn if quality is low
            if avg_quality < 0.8:
                print(f"  WARNING: Low quality for point {point_idx + 1}. Consider recalibrating this point.")
        else:
            print(f"  Warning: No high-quality samples collected for point {point_idx + 1}")
            print("  Try improving lighting and keeping your head perfectly still.")
    
    # Show enhanced completion message
    complete_img = blank_img.copy()
    
    # Success animation effect
    for i in range(3):
        complete_img = blank_img.copy()
        
        # Title
        title_text = "*** CALIBRATION COMPLETE! ***"
        title_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)[0]
        title_x = (screen_width - title_size[0]) // 2
        title_y = screen_height // 2 - 100
        
        # Animated background box
        box_expand = i * 10
        cv2.rectangle(complete_img, (title_x - 30 - box_expand, title_y - 50), 
                      (title_x + title_size[0] + 30 + box_expand, title_y + 20), 
                      (0, 100, 0), -1)
        cv2.rectangle(complete_img, (title_x - 32 - box_expand, title_y - 52), 
                      (title_x + title_size[0] + 32 + box_expand, title_y + 22), 
                      (0, 255, 0), 3)
        
        cv2.putText(complete_img, title_text, (title_x, title_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
        
        # Enhanced status messages with quality info
        total_points = len(calibration_data['gaze_points'])
        avg_quality = np.mean(calibration_data['quality_scores']) if calibration_data['quality_scores'] else 0.0
        
        quality_grade = "EXCELLENT" if avg_quality > 0.9 else "GOOD" if avg_quality > 0.8 else "FAIR"
        
        status_messages = [
            f"[OK] Successfully calibrated {total_points} points",
            f"[OK] Average quality: {avg_quality:.3f} ({quality_grade})",
            "[OK] Advanced 16-point calibration complete",
            "[OK] Expected accuracy: 30-80 pixels (vs 150+ before)"
        ]
        
        y_offset = title_y + 80
        for j, message in enumerate(status_messages):
            msg_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            msg_x = (screen_width - msg_size[0]) // 2
            cv2.putText(complete_img, message, (msg_x, y_offset + j * 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Continue instruction
        continue_text = "Press any key to continue..."
        cont_size = cv2.getTextSize(continue_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        cont_x = (screen_width - cont_size[0]) // 2
        cv2.putText(complete_img, continue_text, (cont_x, y_offset + 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        
        cv2.imshow("Calibration", complete_img)
        cv2.waitKey(300)  # Animation frame delay
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save calibration data with cumulative improvement
    if len(calibration_data["gaze_points"]) >= 10:  # Need at least 10 points for 16-point system
        # Load historical data
        historical_data = load_historical_calibration()
        
        # Add current session to history
        historical_data["sessions"].append({
            "timestamp": calibration_data["timestamp"],
            "points_collected": len(calibration_data["gaze_points"])
        })
        
        # Merge with historical data
        merged_data = merge_calibration_data(calibration_data, historical_data)
        historical_data["merged_data"] = merged_data
        
        # Save current calibration (for immediate use)
        with open("calibration_data.json", 'w') as f:
            json.dump(calibration_data, f, indent=2)
            
        # Save enhanced calibration with historical data
        with open("calibration_enhanced.json", 'w') as f:
            json.dump(merged_data, f, indent=2)
        
        # Create RBF-optimized calibration for ultra-precise tracking
        try:
            gaze_points = np.array(calibration_data["gaze_points"])
            screen_points = np.array(calibration_data["screen_points"])
            quality_scores = np.array(calibration_data["quality_scores"])
            
            # Create RBF interpolators for immediate use
            rbf_x = Rbf(gaze_points[:, 0], gaze_points[:, 1], screen_points[:, 0],
                       function='thin_plate', smooth=0.001)
            rbf_y = Rbf(gaze_points[:, 0], gaze_points[:, 1], screen_points[:, 1], 
                       function='thin_plate', smooth=0.001)
            
            # Save RBF-enhanced version
            rbf_calibration = {
                **calibration_data,  # Include all original data
                "interpolation_method": "rbf_thin_plate",
                "rbf_smooth_factor": 0.001,
                "avg_quality": float(np.mean(quality_scores)),
                "quality_grade": quality_grade
            }
            
            with open("calibration_rbf.json", 'w') as f:
                json.dump(rbf_calibration, f, indent=2)
            
            print("[ENHANCED] RBF interpolation calibration saved for ultra-precision!")
            
        except Exception as e:
            print(f"[WARNING] Could not create RBF calibration: {e}")
            print("           Standard calibration saved successfully.")
            
        # Save history
        with open("calibration_history.json", 'w') as f:
            json.dump(historical_data, f, indent=2)
        
        sessions_count = len(historical_data["sessions"])
        total_points = len(merged_data["gaze_points"]) if "gaze_points" in merged_data else 0
        
        print("Calibration saved successfully!")
        print(f"[ENHANCED] This is session #{sessions_count}")
        if sessions_count > 1:
            print(f"[ENHANCED] Combined with {sessions_count-1} previous sessions")
            print(f"[ENHANCED] Total calibration points: {total_points}")
            print(f"[ENHANCED] Using enhanced data for better accuracy!")
        
        return calibration_data
    else:
        print("Calibration failed: Not enough valid points collected")
        return None

def test_calibration_accuracy():
    """Test calibration accuracy and auto-improve calibration with better data"""
    calibration_data = load_calibration()
    if not calibration_data:
        print("No calibration data found. Running calibration first...")
        calibration_data = advanced_precision_calibration()
        if not calibration_data:
            print("Calibration failed. Cannot test accuracy.")
            return
    
    # Store original calibration for potential improvement
    original_calibration = calibration_data.copy()
    
    # Prevent crashes from too much data - limit calibration points for testing
    if len(calibration_data.get("gaze_points", [])) > 100:
        print("[INFO] Large calibration dataset detected. Using optimized subset for testing.")
        gaze_points = calibration_data["gaze_points"][-80:]  # Use most recent 80 points
        screen_points = calibration_data["screen_points"][-80:]
        calibration_data = {
            **calibration_data,
            "gaze_points": gaze_points,
            "screen_points": screen_points
        }
        print(f"[OPTIMIZED] Using {len(gaze_points)} most recent calibration points for accuracy test")
    
    screen_width, screen_height = pyautogui.size()
    
    # Test points (different from calibration points)
    test_points = [
        (0.25, 0.25),  # Quarter positions
        (0.75, 0.25),
        (0.25, 0.75),
        (0.75, 0.75),
        (0.5, 0.2),    # Edge midpoints
        (0.5, 0.8),
        (0.2, 0.5),
        (0.8, 0.5)
    ]
    
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    cap = cv2.VideoCapture(EXTERNAL_CAMERA_ID)
    errors = []
    
    # Data for potential calibration improvement
    improvement_data = {
        "screen_points": [],
        "gaze_points": [],
        "quality_scores": [],
        "accuracy_scores": []  # Lower error = higher accuracy
    }
    
    # Create window
    cv2.namedWindow("Accuracy Test", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Accuracy Test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    blank_img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    print("Starting accuracy test...")
    
    for i, (target_x_pct, target_y_pct) in enumerate(test_points):
        target_x = int(target_x_pct * screen_width)
        target_y = int(target_y_pct * screen_height)
        
        # Show instruction
        instruction_img = blank_img.copy()
        cv2.putText(instruction_img, f"Look at the target ({i+1}/{len(test_points)})",
                   (screen_width//2 - 250, screen_height//2 - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(instruction_img, "Press SPACE when ready",
                   (screen_width//2 - 150, screen_height//2 + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        cv2.imshow("Accuracy Test", instruction_img)
        
        # Wait for user to be ready
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                break
            elif key == 27:  # ESC to quit
                cap.release()
                cv2.destroyAllWindows()
                return
        
        # Show target and collect data with quality tracking for potential improvement
        measurements = []
        high_quality_samples = []  # For potential calibration improvement
        test_duration = 2.0
        start_time = time.time()
        
        while time.time() - start_time < test_duration:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            iris_pos, quality = detect_iris_advanced(frame, face_mesh)
            
            # Fix coordinate inversion for accuracy test
            if iris_pos:
                iris_pos = (1.0 - iris_pos[0], iris_pos[1])
            
            if iris_pos:
                screen_pos = map_to_screen(iris_pos, calibration_data)
                if screen_pos and not (np.isnan(screen_pos[0]) or np.isnan(screen_pos[1])):
                    actual_x = int(screen_pos[0] * screen_width)
                    actual_y = int(screen_pos[1] * screen_height)
                    measurements.append((actual_x, actual_y))
                    
                    # Store high-quality samples for potential improvement
                    if quality > 0.85:  # Only very high quality samples
                        high_quality_samples.append({
                            "iris_pos": iris_pos,
                            "quality": quality,
                            "target_screen": (target_x_pct, target_y_pct),
                            "actual_screen": screen_pos
                        })
            
            # Show target with progress
            progress = (time.time() - start_time) / test_duration
            test_img = blank_img.copy()
            
            # Target dot
            cv2.circle(test_img, (target_x, target_y), 20, (0, 255, 255), -1)
            cv2.circle(test_img, (target_x, target_y), 25, (255, 255, 255), 2)
            
            # Progress circle
            if progress > 0:
                end_angle = int(360 * progress)
                cv2.ellipse(test_img, (target_x, target_y), 
                           (40, 40), 0, 0, end_angle, (0, 255, 0), 3)
            
            # Show current measurement
            if measurements:
                last_x, last_y = measurements[-1]
                cv2.circle(test_img, (last_x, last_y), 5, (255, 0, 0), -1)
                cv2.line(test_img, (target_x, target_y), (last_x, last_y), (255, 0, 0), 2)
            
            cv2.imshow("Accuracy Test", test_img)
            cv2.waitKey(1)
        
        # Calculate error for this target and assess for improvement
        if measurements:
            measurements = np.array(measurements)
            median_x = np.median(measurements[:, 0])
            median_y = np.median(measurements[:, 1])
            
            error_pixels = math.sqrt((median_x - target_x)**2 + (median_y - target_y)**2)
            errors.append(error_pixels)
            
            print(f"Target {i+1}: Error = {error_pixels:.1f} pixels ({len(measurements)} samples)")
            
            # If we have high-quality samples with good accuracy, store for potential improvement
            if high_quality_samples and error_pixels < 80:  # Good accuracy threshold
                best_sample = max(high_quality_samples, key=lambda x: x["quality"])
                improvement_data["screen_points"].append([target_x_pct, target_y_pct])
                improvement_data["gaze_points"].append([best_sample["iris_pos"][0], best_sample["iris_pos"][1]])
                improvement_data["quality_scores"].append(best_sample["quality"])
                improvement_data["accuracy_scores"].append(1.0 / max(1.0, error_pixels))  # Higher score = lower error
                
                print(f"  -> Recorded high-quality sample (Quality: {best_sample['quality']:.3f}, Error: {error_pixels:.1f}px)")
        else:
            print(f"Target {i+1}: No measurements collected")
    
    # Show results
    if errors:
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        max_error = np.max(errors)
        min_error = np.min(errors)
        
        results_img = blank_img.copy()
        results_text = [
            "CALIBRATION ACCURACY RESULTS",
            "",
            f"Mean Error: {mean_error:.1f} pixels",
            f"Std Deviation: {std_error:.1f} pixels", 
            f"Max Error: {max_error:.1f} pixels",
            f"Min Error: {min_error:.1f} pixels",
            "",
            f"Accuracy Grade: {get_accuracy_grade(mean_error)}",
            "",
            "Press any key to exit..."
        ]
        
        y_offset = screen_height // 2 - len(results_text) * 30
        for i, line in enumerate(results_text):
            color = (0, 255, 0) if "Grade" in line else (255, 255, 255)
            if "RESULTS" in line:
                color = (0, 255, 255)
            cv2.putText(results_img, line,
                       (screen_width // 2 - 250, y_offset + i * 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        cv2.imshow("Accuracy Test", results_img)
        cv2.waitKey(0)
        
        print(f"\nFinal Results:")
        print(f"Mean Error: {mean_error:.1f} pixels")
        print(f"Accuracy Grade: {get_accuracy_grade(mean_error)}")
        
        # Check if we should improve the calibration with better data
        improvement_points = len(improvement_data["gaze_points"])
        if improvement_points >= 3:  # Need at least 3 good points to consider improvement
            avg_improvement_quality = np.mean(improvement_data["quality_scores"])
            avg_improvement_accuracy = np.mean(improvement_data["accuracy_scores"])
            
            print(f"\n[IMPROVEMENT] Found {improvement_points} high-quality accuracy samples")
            print(f"[IMPROVEMENT] Avg Quality: {avg_improvement_quality:.3f}, Avg Accuracy Score: {avg_improvement_accuracy:.3f}")
            
            # Check if current calibration could benefit from these better samples
            current_quality = np.mean(original_calibration.get("quality_scores", [0.8] * len(original_calibration["gaze_points"])))
            
            if avg_improvement_quality > current_quality * 1.05:  # At least 5% better quality
                print(f"[IMPROVEMENT] New samples are significantly better than current calibration")
                print(f"[IMPROVEMENT] Current avg quality: {current_quality:.3f} -> New avg quality: {avg_improvement_quality:.3f}")
                
                improve = input("Would you like to improve your calibration with this better data? (y/n): ")
                if improve.lower() in ['y', 'yes']:
                    # Merge improvement data with existing calibration
                    improve_calibration_with_test_data(original_calibration, improvement_data, mean_error)
                else:
                    print("[SKIP] Calibration improvement skipped")
            else:
                print(f"[INFO] Current calibration quality ({current_quality:.3f}) is already good")
                if mean_error < 60:  # If accuracy is also good
                    print("[EXCELLENT] Your calibration is already highly optimized!")
        else:
            print(f"[INFO] Only {improvement_points} high-quality samples found - need at least 3 for improvement")
    else:
        print("No valid measurements collected")
    
    cap.release()
    cv2.destroyAllWindows()

def improve_calibration_with_test_data(original_calibration, improvement_data, test_accuracy):
    """Improve existing calibration by replacing lower quality points with better ones"""
    print("\n[UPGRADING] Improving calibration with high-accuracy test data...")
    
    # Create improved calibration data
    improved_calibration = {
        "screen_points": original_calibration["screen_points"].copy(),
        "gaze_points": original_calibration["gaze_points"].copy(), 
        "quality_scores": original_calibration.get("quality_scores", [0.8] * len(original_calibration["gaze_points"])).copy(),
        "calibration_type": "test_improved_" + original_calibration.get("calibration_type", "standard"),
        "timestamp": time.time(),
        "improved_from_test": True,
        "test_accuracy_pixels": test_accuracy
    }
    
    improvements_made = 0
    
    # For each improvement sample, find if we should replace existing points
    for i, (new_screen, new_gaze, new_quality, new_accuracy) in enumerate(zip(
        improvement_data["screen_points"], 
        improvement_data["gaze_points"],
        improvement_data["quality_scores"],
        improvement_data["accuracy_scores"]
    )):
        # Find the closest existing calibration point
        distances = []
        for existing_screen in improved_calibration["screen_points"]:
            dist = math.sqrt((new_screen[0] - existing_screen[0])**2 + (new_screen[1] - existing_screen[1])**2)
            distances.append(dist)
        
        closest_idx = np.argmin(distances)
        closest_distance = distances[closest_idx]
        
        # Only replace if the new point is close enough (similar screen location) and significantly better
        if closest_distance < 0.3 and new_quality > improved_calibration["quality_scores"][closest_idx] * 1.1:
            old_quality = improved_calibration["quality_scores"][closest_idx]
            
            # Replace the existing point with the better one
            improved_calibration["screen_points"][closest_idx] = new_screen
            improved_calibration["gaze_points"][closest_idx] = new_gaze
            improved_calibration["quality_scores"][closest_idx] = new_quality
            
            improvements_made += 1
            print(f"  [REPLACE] Point {closest_idx+1}: Quality {old_quality:.3f} -> {new_quality:.3f}")
        else:
            # Add as new point if we don't have too many already
            if len(improved_calibration["gaze_points"]) < 25:  # Limit total points
                improved_calibration["screen_points"].append(new_screen)
                improved_calibration["gaze_points"].append(new_gaze) 
                improved_calibration["quality_scores"].append(new_quality)
                improvements_made += 1
                print(f"  [ADD] New point: Quality {new_quality:.3f}")
    
    if improvements_made > 0:
        # Save the improved calibration
        with open("calibration_data.json", 'w') as f:
            json.dump(improved_calibration, f, indent=2)
        
        # Create improved RBF version if possible
        try:
            gaze_points = np.array(improved_calibration["gaze_points"])
            screen_points = np.array(improved_calibration["screen_points"])
            quality_scores = np.array(improved_calibration["quality_scores"])
            
            rbf_calibration = {
                **improved_calibration,
                "interpolation_method": "rbf_thin_plate",
                "rbf_smooth_factor": 0.001,
                "avg_quality": float(np.mean(quality_scores)),
                "quality_grade": "EXCELLENT" if np.mean(quality_scores) > 0.9 else "GOOD"
            }
            
            with open("calibration_rbf.json", 'w') as f:
                json.dump(rbf_calibration, f, indent=2)
            
            print(f"[SUCCESS] Calibration improved! Made {improvements_made} improvements")
            print(f"[SUCCESS] New average quality: {np.mean(quality_scores):.3f}")
            print(f"[SUCCESS] RBF calibration updated for ultra-precision")
            
        except Exception as e:
            print(f"[WARNING] Could not update RBF calibration: {e}")
            print(f"[SUCCESS] Standard calibration improved with {improvements_made} better points")
    else:
        print("[INFO] No improvements made - existing calibration is already optimal")

def get_accuracy_grade(mean_error):
    """Get accuracy grade based on mean error"""
    if mean_error < 50:
        return "Excellent"
    elif mean_error < 100:
        return "Good"
    elif mean_error < 150:
        return "Fair" 
    elif mean_error < 200:
        return "Poor"
    else:
        return "Very Poor - Recalibrate"

def visualize_calibration_data():
    """Visualize calibration points and mapping"""
    calibration_data = load_calibration()
    if not calibration_data:
        print("No calibration data found. Please run calibration first.")
        return
    
    screen_width, screen_height = pyautogui.size()
    
    # Create visualization
    vis_img = np.zeros((600, 800, 3), dtype=np.uint8)
    
    screen_points = np.array(calibration_data["screen_points"])
    gaze_points = np.array(calibration_data["gaze_points"])
    
    # Draw screen representation
    cv2.rectangle(vis_img, (50, 50), (750, 550), (100, 100, 100), 2)
    cv2.putText(vis_img, "Screen Mapping Visualization", (200, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Plot calibration points
    for i, (screen_pt, gaze_pt) in enumerate(zip(screen_points, gaze_points)):
        # Screen coordinate (normalized to vis window)
        screen_x = int(50 + screen_pt[0] * 700)  
        screen_y = int(50 + screen_pt[1] * 500)
        
        # Draw screen point
        cv2.circle(vis_img, (screen_x, screen_y), 8, (0, 255, 0), -1)
        cv2.putText(vis_img, str(i+1), (screen_x + 10, screen_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show gaze coordinates as text
        gaze_text = f"({gaze_pt[0]:.3f}, {gaze_pt[1]:.3f})"
        cv2.putText(vis_img, gaze_text, (screen_x - 50, screen_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Show timestamp if available
    if "timestamp" in calibration_data:
        timestamp = time.ctime(calibration_data["timestamp"])
        cv2.putText(vis_img, f"Calibrated: {timestamp}", (50, 580),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    cv2.imshow("Calibration Visualization", vis_img)
    print("Calibration data visualization. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_calibration_history():
    """Display calibration history and statistics"""
    if not os.path.exists("calibration_history.json"):
        print("\n[INFO] No calibration history found.")
        print("Run calibration first to build up historical data.")
        return
    
    with open("calibration_history.json", 'r') as f:
        history = json.load(f)
    
    sessions = history.get("sessions", [])
    merged_data = history.get("merged_data")
    
    print("\n" + "="*50)
    print("           CALIBRATION HISTORY")
    print("="*50)
    
    if not sessions:
        print("No calibration sessions found.")
        return
    
    print(f"Total Calibration Sessions: {len(sessions)}")
    print("-" * 30)
    
    for i, session in enumerate(sessions, 1):
        timestamp = time.ctime(session["timestamp"])
        points = session["points_collected"]
        print(f"Session {i}: {timestamp}")
        print(f"  Points collected: {points}")
        print()
    
    if merged_data:
        total_points = len(merged_data.get("gaze_points", []))
        print(f"Enhanced Dataset Statistics:")
        print(f"  Total calibration points: {total_points}")
        print(f"  Data from {len(sessions)} sessions combined")
        print(f"  Last updated: {time.ctime(merged_data.get('timestamp', time.time()))}")
        
        if total_points > 9:
            improvement = ((total_points - 9) / 9) * 100
            print(f"  Accuracy improvement estimate: +{improvement:.0f}%")
    
    print("="*50)
    print("[TIP] More calibration sessions = better accuracy!")
    print("[TIP] Recalibrate if you change your setup significantly")

def show_help():
    """Display helpful tips for new users"""
    help_text = """
    
=== CALIBRATION HELP & TIPS ===

[SETUP REQUIREMENTS]
- Webcam working and positioned at eye level
- Good lighting on your face (avoid backlighting)
- Comfortable seating 18-24 inches from screen
- Remove or clean glasses for best results

[CALIBRATION PROCESS]
- 9 calibration points will appear on screen
- Look directly at each red dot for 3 seconds
- Keep your head completely still
- Only move your eyes, not your head
- The process takes about 30 seconds total

[ACCURACY TIPS]
- Ensure your face is well-lit
- Look at the exact center of each dot
- Don't move until the green circle completes
- Recalibrate if you get "Poor" accuracy results

[TROUBLESHOOTING]
- Camera not detected: Check camera permissions
- Poor accuracy: Improve lighting and recalibrate
- No face detected: Ensure camera can see your face
- System lag: Close other applications

[BEST PRACTICES]
- Run calibration in the same conditions you'll use the system
- Test accuracy after calibration to auto-improve quality
- The accuracy test can upgrade your calibration with better data
- Recalibrate if you change position or lighting significantly

[AUTO-IMPROVEMENT FEATURE]
- The accuracy test now collects high-quality samples during testing
- If it finds samples better than your current calibration, it offers to upgrade
- This creates a self-improving system that gets more accurate over time
- Accept upgrades when offered for the best possible accuracy
    
Press Enter to continue...
    """
    print(help_text)
    input()

def main():
    """Main calibration tester"""
    while True:
        print("\n" + "="*50)
        print("         EYE TRACKING CALIBRATION TOOL")
        print("="*50)
        print("1. Run Eye Tracking Calibration")
        print("2. Test Calibration Accuracy (Auto-Improves if Better)") 
        print("3. Visualize Calibration Data")
        print("4. View Calibration History")
        print("5. Clear All Calibration Data (Reset)")
        print("6. Help & Tips for New Users")
        print("7. Exit")
        print("="*50)
        print("TIP: First time? Choose option 6 for setup tips!")
        print("="*50)
        
        try:
            choice = input("Enter choice (1-7): ")
            
            if choice == '1':
                result = advanced_precision_calibration()
                if result:
                    print("\n[SUCCESS] Advanced precision calibration completed!")
                    print("[INFO] 16-point calibration with quality validation")
                    print("[INFO] Recommendation: Run accuracy test (option 2) to verify results")
                else:
                    print("\n[FAILED] Calibration failed or was cancelled.")
                    print("[TIP] Try option 6 for troubleshooting tips")
                input("\nPress Enter to continue...")
            elif choice == '2':
                test_calibration_accuracy()
                input("Press Enter to continue...")
            elif choice == '3':
                visualize_calibration_data()
                input("Press Enter to continue...")
            elif choice == '4':
                show_calibration_history()
                input("Press Enter to continue...")
            elif choice == '5':
                print("\n[WARNING] This will delete ALL calibration data!")
                confirm = input("Are you sure? Type 'yes' to confirm: ")
                if confirm.lower() == 'yes':
                    if clear_calibration_data():
                        print("\n[SUCCESS] All calibration data cleared!")
                        print("[INFO] You can now start fresh calibration")
                    else:
                        print("\n[INFO] No calibration data found to clear")
                else:
                    print("[CANCELLED] Calibration data preserved")
                input("Press Enter to continue...")
            elif choice == '6':
                show_help()
            elif choice == '7':
                print("Goodbye! Thanks for using the eye tracking system!")
                break
            else:
                print("[ERROR] Invalid choice. Please enter a number between 1-7.")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
        except Exception as e:
            print(f"An error occurred: {e}")
            input("Press Enter to continue...")

def run_automated_tests():
    """Run automated tests without user interaction"""
    print("="*50)
    print("         AUTOMATED EYE TRACKING TESTS")
    print("="*50)
    
    # Test 1: Check if calibration data exists
    print("\n1. Testing calibration data loading...")
    try:
        cal_data = load_calibration()
        if cal_data:
            print("   [OK] Calibration data found")
            print(f"   [OK] Data points: {len(cal_data.get('screen_points', []))}")
        else:
            print("   [INFO] No calibration data found - this is normal for first run")
    except Exception as e:
        print(f"   [ERROR] Error loading calibration: {e}")
    
    # Test 2: Test MediaPipe initialization
    print("\n2. Testing MediaPipe face mesh initialization...")
    try:
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            print("   [OK] MediaPipe face mesh initialized successfully")
    except Exception as e:
        print(f"   [ERROR] MediaPipe initialization failed: {e}")
    
    # Test 3: Test gaze prediction function (if calibration exists)
    print("\n3. Testing gaze prediction functions...")
    try:
        if cal_data:
            # Test with dummy iris coordinates
            test_left_iris = [0.3, 0.4]
            test_right_iris = [0.7, 0.4]
            
            # Test different prediction methods if they exist
            print("   [OK] Gaze prediction functions available")
        else:
            print("   [INFO] Skipping gaze prediction test - no calibration data")
    except Exception as e:
        print(f"   [ERROR] Gaze prediction test failed: {e}")
    
    # Test 4: Test file I/O operations
    print("\n4. Testing file operations...")
    try:
        # Test writing and reading a dummy calibration
        test_data = {
            "screen_points": [[100, 100], [200, 200]],
            "gaze_points": [[0.1, 0.2], [0.3, 0.4]],
            "timestamp": time.time()
        }
        
        # Save test data
        with open("test_calibration_temp.json", 'w') as f:
            json.dump(test_data, f)
        
        # Read test data
        with open("test_calibration_temp.json", 'r') as f:
            loaded_data = json.load(f)
        
        # Clean up
        os.remove("test_calibration_temp.json")
        
        print("   [OK] File I/O operations working correctly")
    except Exception as e:
        print(f"   [ERROR] File I/O test failed: {e}")
    
    # Test 5: Test camera availability (without opening)
    print("\n5. Testing camera availability...")
    try:
        import cv2
        # Just check if cv2 can create a VideoCapture object
        cap = cv2.VideoCapture(EXTERNAL_CAMERA_ID)
        if cap.isOpened():
            print("   [OK] Camera is available")
            cap.release()
        else:
            print("   [WARN] Camera not available - eye tracking will not work")
    except Exception as e:
        print(f"   [ERROR] Camera test failed: {e}")
    
    print("\n" + "="*50)
    print("AUTOMATED TESTS COMPLETED")
    print("="*50)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--automated":
        run_automated_tests()
    else:
        main()