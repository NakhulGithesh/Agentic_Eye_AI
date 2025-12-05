try:
    import mediapipe
    print('MediaPipe is installed')
    print(f'MediaPipe version: {mediapipe.__version__}')
except ImportError as e:
    print(f'Error: {e}')

try:
    import cv2
    print('OpenCV is installed')
    print(f'OpenCV version: {cv2.__version__}')
except ImportError as e:
    print(f'Error: {e}')

try:
    import numpy
    print('NumPy is installed')
    print(f'NumPy version: {numpy.__version__}')
except ImportError as e:
    print(f'Error: {e}')

try:
    import pyautogui
    print('PyAutoGUI is installed')
    print(f'PyAutoGUI version: {pyautogui.__version__}')
except ImportError as e:
    print(f'Error: {e}')

try:
    import pynput
    print('Pynput is installed')
    # pynput doesn't have a __version__ attribute
    print('Pynput version: Not available through __version__')
except ImportError as e:
    print(f'Error: {e}')

try:
    import scipy
    print('SciPy is installed')
    print(f'SciPy version: {scipy.__version__}')
except ImportError as e:
    print(f'Error: {e}')