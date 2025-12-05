import os
import subprocess
import sys
import time

try:
    from modules.mirroring.scrcpy_manager import ScrcpyManager
    from config.settings import SETTINGS
    MIRRORING_AVAILABLE = True
except ImportError:
    MIRRORING_AVAILABLE = False
    print("Warning: Mirroring module not available")

def clear_screen():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def find_external_camera():
    """Find external phone camera (usually index 1 for external webcam apps)"""
    try:
        import cv2
        # Check cameras in order of preference for external phone
        for camera_id in [1, 2, 0]:  # Try external phone first (1), then other options
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                # Test if we can read a frame
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    print(f"+ External Camera found at index: {camera_id}")
                    return camera_id
        print("- No suitable external camera found, using default")
        return 1  # Default to index 1 for external phone
    except:
        return 1

def check_dependencies():
    """Check if all required dependencies are installed"""
    missing_deps = []
    
    try:
        import cv2
        print(f"+ OpenCV: {cv2.__version__}")
    except ImportError:
        missing_deps.append("opencv-python")
        print("- OpenCV: NOT FOUND")

    try:
        import mediapipe
        print(f"+ MediaPipe: {mediapipe.__version__}")
    except ImportError:
        missing_deps.append("mediapipe")
        print("- MediaPipe: NOT FOUND")

    try:
        import numpy
        print(f"+ NumPy: {numpy.__version__}")
    except ImportError:
        missing_deps.append("numpy")
        print("- NumPy: NOT FOUND")                                                                                                                                                                                                                                                             

    try:
        import pyautogui
        print(f"+ PyAutoGUI: {pyautogui.__version__}")
    except ImportError:
        missing_deps.append("pyautogui")
        print("- PyAutoGUI: NOT FOUND")

    try:
        import pynput
        print("+ Pynput: Available")
    except ImportError:
        missing_deps.append("pynput")
        print("- Pynput: NOT FOUND")

    try:
        import scipy
        print(f"+ SciPy: {scipy.__version__}")
    except ImportError:
        missing_deps.append("scipy")
        print("- SciPy: NOT FOUND")

    try:
        import psutil
        print(f"+ Psutil: {psutil.__version__}")
    except ImportError:
        missing_deps.append("psutil")
        print("- Psutil: NOT FOUND")
    
    # Test external phone webcam
    try:
        import cv2
        camera_id = find_external_camera()
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            print(f"+ External Phone Webcam: Available at index {camera_id}")
            cap.release()
        else:
            print("- External Phone Webcam: Not accessible")
            return False
    except:
        print("- External Phone Webcam: Error accessing")
        return False
    
    # Test Android mirroring capabilities
    if MIRRORING_AVAILABLE:
        try:
            manager = ScrcpyManager()
            status = manager.get_status()
            if status['device_connected']:
                print(f"+ Android Device: Connected ({len(status['connected_devices'])} device(s))")
                print(f"+ Screen Mirroring: Available via scrcpy")
            else:
                print("- Android Device: Not connected via ADB")
        except Exception as e:
            print(f"- Screen Mirroring: Error - {e}")
    else:
        print("- Screen Mirroring: Module not available")
    
    if missing_deps:
        print(f"\nMissing dependencies: {', '.join(missing_deps)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\nAll dependencies are available!")
    return True

def display_menu():
    """Display the main menu"""
    clear_screen()
    print("=" * 50)
    print("           AGENTIC EYE LAUNCHER")
    print("=" * 50)
    print("1. Install/Check Dependencies")
    print("2. Test Iris Tracking")
    print("3. Run Standard Calibration & Test")
    print("4. Run IMPROVED Calibration (Recommended)")
    print("5. Run Full Eye Control System + Mirroring")
    print("6. Start/Stop Android Screen Mirroring")
    print("7. Exit")
    print("=" * 50)
    print("NEW: Option 5 now auto-starts Android mirroring!")
    print("NOTE: Use IMPROVED calibration (option 4) for best results!")
    print("=" * 50)

def run_script(script_name):
    """Run a Python script"""
    print(f"\nStarting {script_name}...")
    print("-" * 50)
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, text=True)
        if result.returncode == 0:
            print(f"\n{script_name} completed successfully!")
        else:
            print(f"\n{script_name} exited with code {result.returncode}")
    except Exception as e:
        print(f"Error running {script_name}: {e}")
    
    input("\nPress Enter to continue...")

def install_dependencies():
    """Install dependencies from requirements.txt"""
    print("\nInstalling dependencies...")
    print("-" * 30)
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        if result.returncode == 0:
            print("Dependencies installed successfully!")
        else:
            print("Error installing dependencies")
    except Exception as e:
        print(f"Error: {e}")
    
    input("Press Enter to continue...")

def check_calibration_file():
    """Check if calibration file exists"""
    if os.path.exists("calibration_data.json"):
        print("Calibration file found!")
        return True
    else:
        print("No calibration file found. You need to run calibration first.")
        return False

def start_mirroring():
    """Start Android screen mirroring"""
    if not MIRRORING_AVAILABLE:
        print("Mirroring module not available!")
        return False
    
    try:
        manager = ScrcpyManager()
        status = manager.get_status()
        
        if not status['device_connected']:
            print("No Android device connected via ADB!")
            print("Please connect your Samsung S8 via USB and enable USB debugging.")
            return False
        
        if status['is_running']:
            print("Screen mirroring is already running!")
            return True
        
        print("Starting Android screen mirroring...")
        print(f"Connected devices: {status['connected_devices']}")
        
        if manager.start_mirroring():
            print("✓ Screen mirroring started successfully!")
            print("  Window title: 'AgenticEye Mirror'")
            time.sleep(2)
            return True
        else:
            print("✗ Failed to start screen mirroring")
            return False
            
    except Exception as e:
        print(f"Error starting mirroring: {e}")
        return False

def run_full_eye_control_with_mirroring():
    """Run the full eye control system with automatic mirroring"""
    print("\n" + "="*50)
    print("    STARTING FULL EYE CONTROL SYSTEM")
    print("="*50)
    
    # Start mirroring first
    print("\n1. Initializing Android Screen Mirroring...")
    mirroring_started = start_mirroring()
    
    if mirroring_started:
        print("\n2. Starting Eye Control System...")
        print("-" * 50)
        time.sleep(1)
        
        try:
            result = subprocess.run([sys.executable, "precise_eye_controller.py"], 
                                  capture_output=False, text=True)
            if result.returncode == 0:
                print("\nEye Control System completed successfully!")
            else:
                print(f"\nEye Control System exited with code {result.returncode}")
        except Exception as e:
            print(f"Error running eye control system: {e}")
    else:
        print("\nContinuing without mirroring...")
        proceed = input("Start eye control system anyway? (y/n): ")
        if proceed.lower() in ['y', 'yes']:
            run_script("precise_eye_controller.py")
    
    input("\nPress Enter to continue...")

def main():
    """Main function"""
    while True:
        display_menu()
        
        try:
            choice = input("Enter your choice (1-7): ")
            
            if choice == '1':
                clear_screen()
                print("Checking dependencies...")
                print("-" * 30)
                deps_ok = check_dependencies()
                if not deps_ok:
                    install_deps = input("\nInstall missing dependencies? (y/n): ")
                    if install_deps.lower() in ['y', 'yes']:
                        install_dependencies()
                else:
                    input("\nPress Enter to continue...")
                    
            elif choice == '2':
                if check_dependencies():
                    run_script("test_iris_tracking.py")
                else:
                    print("Please install dependencies first!")
                    input("Press Enter to continue...")
                    
            elif choice == '3':
                if check_dependencies():
                    run_script("test_calibration.py")
                else:
                    print("Please install dependencies first!")
                    input("Press Enter to continue...")
                    
            elif choice == '4':
                if check_dependencies():
                    run_script("improved_calibration.py")
                else:
                    print("Please install dependencies first!")
                    input("Press Enter to continue...")
                    
            elif choice == '5':
                if not check_dependencies():
                    print("Please install dependencies first!")
                    input("Press Enter to continue...")
                elif not check_calibration_file():
                    run_cal = input("Run calibration first? (y/n): ")
                    if run_cal.lower() in ['y', 'yes']:
                        run_script("improved_calibration.py")
                    else:
                        input("Press Enter to continue...")
                else:
                    run_full_eye_control_with_mirroring()
                    
            elif choice == '6':
                if not MIRRORING_AVAILABLE:
                    print("Mirroring module not available!")
                    input("Press Enter to continue...")
                else:
                    clear_screen()
                    print("Android Screen Mirroring Control")
                    print("-" * 30)
                    try:
                        manager = ScrcpyManager()
                        status = manager.get_status()
                        
                        print(f"Device connected: {status['device_connected']}")
                        if status['connected_devices']:
                            print(f"Connected devices: {status['connected_devices']}")
                        print(f"Mirroring running: {status['is_running']}")
                        
                        if status['is_running']:
                            stop = input("\nStop mirroring? (y/n): ")
                            if stop.lower() in ['y', 'yes']:
                                if manager.stop_mirroring():
                                    print("Mirroring stopped successfully!")
                                else:
                                    print("Failed to stop mirroring")
                        else:
                            start = input("\nStart mirroring? (y/n): ")
                            if start.lower() in ['y', 'yes']:
                                start_mirroring()
                    except Exception as e:
                        print(f"Error: {e}")
                    
                    input("Press Enter to continue...")
                    
            elif choice == '7':
                clear_screen()
                print("Thank you for using Agentic Eye!")
                break
            else:
                print("Invalid choice. Please enter 1-7.")
                input("Press Enter to continue...")
                
        except KeyboardInterrupt:
            clear_screen()
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()