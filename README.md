# Agentic Eye AI - Phone Mirror with Eye Tracking Control

## 🎯 Overview

Agentic Eye AI is an innovative accessibility application that combines Android screen mirroring with advanced eye tracking technology. The system allows users to control their Android devices using natural eye movements, making technology more accessible for individuals with motor disabilities or those who prefer hands-free interaction.

The application provides a comprehensive control interface that mirrors your Android device's screen and enables precise eye-tracking based navigation and interaction.

## ✨ Key Features

- **Real-time Android Screen Mirroring**: Mirror your Android device screen using scrcpy
- **Advanced Eye Tracking Control**: Professional-grade eye tracking with 25-point calibration
- **Intuitive Control Panels**: Three-panel interface for comprehensive device control
- **Accessibility Focused**: Designed for hands-free operation
- **High-Precision Calibration**: Enhanced calibration system with outlier rejection and RBF interpolation
- **Real-time Performance Monitoring**: Live feedback on tracking accuracy and system status

## 📋 System Requirements

### Hardware Requirements
- **Computer**: Windows 10/11 with modern CPU (i5/i7 recommended)
- **Webcam**: HD webcam (720p or higher) with good autofocus
- **Android Device**: Android 5.0+ with USB debugging enabled
- **USB Cable**: High-quality USB cable for reliable connection

### Software Requirements
- **Python**: 3.7 or higher
- **ADB**: Android Debug Bridge (included in platform-tools)
- **Scrcpy**: Screen mirroring tool (included)

### Environmental Requirements
- **Lighting**: Good, even lighting on the user's face
- **Distance**: 50-70cm from webcam
- **Stability**: Minimal head movement during use

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/NakhulGithesh/IHNA-WEB.git Agentic_Eye_AI
cd Agentic_Eye_AI
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

For the eye tracking system, additional dependencies are needed:
```bash
pip install mediapipe==0.10.21 opencv-python==4.8.0.76 numpy==1.24.3 pyautogui==0.9.54 pynput==1.7.6 scipy==1.11.2
```

### 3. Connect Android Device
1. Enable **Developer Options** on your Android device
2. Enable **USB Debugging**
3. Connect your device via USB cable
4. Allow USB debugging authorization when prompted

## 🚀 Usage Guide

### Starting the Application
```bash
python _main.py
```

This launches the Phone Mirror application with the three-panel control interface.

### Panel Overview

#### Left Control Panel
- **Light Control**: Toggle device flashlight
- **Speaker Control**: Audio output controls
- **Wheelchair Control**: Specialized accessibility controls
- **My Phone**: Toggle screen mirroring on/off
- **Close Menu**: Exit the application

#### Center Phone Area
- Displays the mirrored Android screen
- Embedded scrcpy window for real-time mirroring
- Visual feedback for connection status

#### Right Control Panel
- **Volume Controls**: Up/Down volume adjustment
- **Scroll Controls**: Page scrolling functionality
- **Navigation**: Home, Back, and Interact buttons
- **Device Status**: Connection and mirroring status

### Eye Tracking Setup

1. **Navigate to Eye_code directory**:
   ```bash
   cd Eye_code
   ```

2. **Run Enhanced Calibration**:
   ```bash
   python enhanced_calibration.py
   ```
   - Follow the on-screen instructions
   - Maintain stable head position
   - Look at each calibration point for 4 seconds
   - Complete all 25 calibration points

3. **Start Eye Tracking Control**:
   ```bash
   python precise_eye_controller.py
   ```

### Eye Tracking Controls
- **Gaze Movement**: Cursor follows your eye movements
- **Double Blink**: Left mouse click
- **SPACE**: Pause/Resume tracking
- **D**: Toggle debug display
- **R**: Recalibrate system
- **Q**: Quit eye tracking

## 🔧 Configuration

### Application Settings
Modify settings in the following files:
- `ui/_styles.py`: Interface appearance and colors
- `ui/_control_panels.py`: Control panel behavior
- `modules/mirroring/scrcpy_manager.py`: Mirroring configuration

### Eye Tracking Parameters
Adjust parameters in the eye tracking modules:
- `movement_threshold`: Minimum cursor movement distance
- `smoothing_factor`: Mouse movement smoothing (0-1)
- `blink_threshold`: Blink detection sensitivity
- `calibration_duration`: Seconds per calibration point

## 🎯 Performance Optimization

### For Best Eye Tracking Results:
1. **Consistent Environment**: Same lighting and position as calibration
2. **Quality Webcam**: Higher resolution improves tracking accuracy
3. **Stable Setup**: Minimize vibrations and background movement
4. **Regular Recalibration**: Recalibrate weekly or when accuracy decreases

### Expected Performance Metrics:
- **Accuracy**: 30-50 pixel average error
- **Responsiveness**: Smooth cursor tracking
- **Reliability**: Consistent interaction detection

## 🔍 Troubleshooting

### Common Issues

#### "No Device Connected"
- Verify USB cable connection
- Enable USB debugging in Developer Options
- Try different USB ports or cable
- Restart ADB: `adb kill-server && adb start-server`

#### Poor Eye Tracking Accuracy
- Recalibrate in current lighting conditions
- Ensure consistent head position
- Check webcam focus and cleanliness
- Adjust room lighting for better face detection

#### Mirroring Won't Start
- Verify scrcpy and ADB paths in code
- Check Android device compatibility
- Ensure no other scrcpy instances are running
- Update USB drivers if needed

#### Eye Tracking Not Detected
- Confirm webcam permissions
- Test with different lighting conditions
- Check for interfering applications
- Verify all Python dependencies are installed

## 🏗️ Project Structure

```
Agentic_Eye_AI/
├── _main.py                    # Main application launcher
├── requirements.txt            # Python dependencies
├── ui/                         # User interface components
│   ├── _main_window.py         # Main window with three-panel layout
│   ├── _control_panels.py      # Left and right control panels
│   └── _styles.py              # UI styling and constants
├── modules/                    # Core functionality modules
│   └── mirroring/
│       └── scrcpy_manager.py   # Scrcpy integration
├── Eye_code/                   # Eye tracking system
│   ├── enhanced_calibration.py # 25-point calibration system
│   ├── precise_eye_controller.py # Advanced eye tracking controller
│   ├── platform-tools/         # ADB and scrcpy binaries
│   └── config/                 # Eye tracking configuration
└── README.md                   # This file
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Enhanced accessibility features
- Additional device control options
- Improved eye tracking algorithms
- Cross-platform compatibility
- User interface enhancements

## 📄 License

This project is developed for accessibility purposes. Please refer to individual component licenses for distribution terms.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review console output for error messages
3. Ensure all dependencies are properly installed
4. Test with known working hardware configuration

## 🔄 Future Enhancements

- **Gesture Recognition**: Additional eye movement patterns
- **Voice Integration**: Combined eye and voice control
- **Multi-Device Support**: Control multiple devices simultaneously
- **Cloud Synchronization**: Save calibration profiles
- **Advanced Analytics**: Usage tracking and performance metrics

---

**Agentic Eye AI** represents a significant step forward in assistive technology, combining cutting-edge computer vision with practical accessibility solutions to empower users with intuitive, hands-free device control.
