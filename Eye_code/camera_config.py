"""
Camera configuration for external phone webcam
"""
import cv2

def get_external_camera_id():
    """
    Find and return the camera ID for external phone webcam.
    
    Returns:
        int: Camera ID (usually 1 for external phone webcam apps like DroidCam, EpocCam, etc.)
    """
    # Check cameras in order of preference for external phone
    for camera_id in [1, 2, 0]:  # Try external phone first (1), then other options
        try:
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                # Test if we can read a frame with timeout
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None and frame.size > 0:
                    print(f"Using external camera at index: {camera_id}")
                    return camera_id
        except Exception as e:
            continue
    
    # Default to index 1 for external phone (most common for external webcam apps)
    print("Using default external camera index: 1")
    return 1

# Global camera ID for external phone
EXTERNAL_CAMERA_ID = get_external_camera_id()