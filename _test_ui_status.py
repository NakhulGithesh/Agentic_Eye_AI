#!/usr/bin/env python3
"""
Test script for UI status updates
"""

import sys
import os
from PyQt5.QtWidgets import QApplication

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import UI components
from ui import MainWindow

def test_ui_updates():
    app = QApplication(sys.argv)
    
    # Create window
    window = MainWindow()
    window.show()
    
    # Test status updates directly
    print("Testing UI status updates...")
    
    # Test right panel update
    print("Testing right panel connection status...")
    window.right_panel.update_connection_status(True, 1)
    
    # Test center area update  
    print("Testing center area status...")
    window.center_area.update_mirror_status("TEST: Device connected via test script")
    
    # Test left panel update
    print("Testing left panel status...")
    window.left_panel.update_status("TEST: Status updated")
    
    # Force device status update
    print("Testing device status update...")
    if window.scrcpy_manager:
        try:
            connected = window.scrcpy_manager.check_device_connected()
            devices = window.scrcpy_manager.get_connected_devices()
            print(f"Scrcpy manager found - Connected: {connected}, Devices: {devices}")
            
            # Force UI update
            window.update_device_status()
            
        except Exception as e:
            print(f"Error with scrcpy manager: {e}")
    else:
        print("No scrcpy manager found")
    
    print("UI test completed. Check if status appears in the application window.")
    print("Close the window to exit.")
    
    # Run the app
    app.exec_()

if __name__ == "__main__":
    test_ui_updates()