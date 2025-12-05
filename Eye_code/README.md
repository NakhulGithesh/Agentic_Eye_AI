# Enhanced Precise Eye Tracking System

## 🎯 Overview
This enhanced system provides precise eye tracking with improved calibration quality and accurate mouse control. The system uses advanced interpolation techniques and better filtering for professional-grade eye tracking.

## 📋 System Requirements
- Python 3.7 or higher
- Webcam with good resolution (720p or higher recommended)
- Good, consistent lighting
- Stable head position during use

## 📦 Installation

### 1. Install Dependencies
```bash
pip install mediapipe==0.10.21
pip install opencv-python==4.8.0.76
pip install numpy==1.24.3
pip install pyautogui==0.9.54
pip install pynput==1.7.6
pip install scipy==1.11.2
```

### 2. File Structure
Your project should have these files:
```
eye_tracking_system/
├── enhanced_calibration.py     (New enhanced calibration)
├── precise_eye_controller.py   (New precise controller)
├── launcher.py                 (Keep existing)
├── test_iris_tracking.py      (Keep existing)
├── requirements.txt            (Updated)
└── README.md                   (Updated guide)
```

## 🚀 Usage Instructions

### Step 1: Run Enhanced Calibration
```bash
python enhanced_calibration.py
```

**Calibration Tips:**
- Sit 50-70cm from your webcam
- Ensure good, even lighting on your face
- Keep your head completely still during calibration
- Look directly at each red dot with your eyes only
- The system will collect 25 calibration points for maximum accuracy
- Each point takes 4 seconds - be patient for best results

### Step 2: Run the Precise Eye Controller
```bash
python precise_eye_controller.py
```

**Usage Tips:**
- Maintain the same head position as during calibration
- Double blink (two quick blinks) to perform left click
- Use keyboard shortcuts for control

## ⌨️ Keyboard Controls

### During Eye Tracking:
- `SPACE` - Toggle pause/resume tracking
- `D` - Toggle debug display on/off
- `R` - Recalibrate system
- `Q` - Quit application

### During Calibration:
- `SPACE` - Start calibration process
- `ESC` - Cancel calibration

## 🔧 Key Improvements

### Enhanced Calibration System:
- **25-point calibration** instead of 9 points
- **Quality-based sample collection** - only accepts high-quality gaze data
- **Outlier rejection** using statistical filtering
- **Weighted averaging** based on sample quality
- **RBF interpolation** for smooth, accurate mapping

### Precise Eye Controller:
- **Advanced temporal filtering** for stability
- **Micro-movement suppression** to reduce jitter
- **Adaptive smoothing** based on movement distance
- **Improved blink detection** with cooldown periods
- **Real-time performance monitoring**

## 🎯 Expected Performance

After proper calibration, you should achieve:
- **Accuracy**: 30-50 pixel average error (Excellent grade)
- **Responsiveness**: Smooth cursor movement following your gaze
- **Reliability**: Consistent click detection with double blinks
- **Stability**: Minimal cursor jitter during fixations

## 🔍 Troubleshooting

### Poor Calibration Quality:
1. **Check lighting** - ensure even, bright lighting on your face
2. **Camera position** - webcam should be at eye level
3. **Head stability** - keep head completely still during calibration
4. **Distance** - maintain consistent 50-70cm distance
5. **Recalibrate** - run calibration multiple times if needed

### Inaccurate Mouse Movement:
1. **Recalibrate** - ensure you're in the same position as during calibration
2. **Check debug info** - enable debug display to see tracking data
3. **Adjust thresholds** - modify movement_threshold in code if needed
4. **Camera settings** - ensure good webcam resolution and frame rate

### Inconsistent Clicking:
1. **Blink clearly** - make distinct double blinks
2. **Timing** - allow 0.2-0.7 seconds between blinks
3. **Wait for cooldown** - system has built-in click prevention
4. **Check EAR values** - monitor in debug display

## 🔧 Advanced Configuration

You can modify these parameters in the code:

### In PreciseEyeController:
```python
self.movement_threshold = 12        # Minimum pixels to move cursor
self.smoothing_factor = 0.25       # Mouse movement smoothing (0-1)
self.blink_threshold = 0.23        # Blink detection sensitivity
```

### In Enhanced Calibration:
```python
calibration_duration = 4.0         # Seconds per calibration point
quality_threshold = 0.7            # Minimum sample quality
```

## 📊 Performance Monitoring

The system provides real-time feedback:
- **FPS counter** - tracking performance
- **Gaze coordinates** - current eye position
- **Mouse position** - actual cursor location
- **EAR values** - blink detection status
- **Calibration status** - system readiness

## 🆘 Common Issues

### "No calibration found" error:
- Run `python enhanced_calibration.py` first
- Ensure calibration completes successfully
- Check for `precise_calibration.json` file

### Poor tracking accuracy:
- Recalibrate in the same lighting conditions
- Ensure consistent head position
- Check webcam focus and cleanliness

### Mouse not moving:
- Verify calibration loaded successfully
- Check debug display for gaze detection
- Ensure tracking is not paused (SPACE key)

## 📈 Performance Optimization

For best results:
1. **Consistent environment** - same lighting, position, and setup
2. **Quality webcam** - higher resolution provides better tracking
3. **Stable setup** - minimize vibrations and movement
4. **Regular recalibration** - recalibrate if you change position
5. **Optimal distance** - maintain 50-70cm from camera

## 🔄 Maintenance

- **Recalibrate weekly** or when accuracy decreases
- **Clean camera lens** regularly for clear image
- **Update dependencies** periodically for best performance
- **Monitor system resources** - close unnecessary applications

This enhanced system should provide significantly better accuracy and reliability compared to the original version. The 25-point calibration and advanced filtering techniques ensure professional-grade eye tracking performance.