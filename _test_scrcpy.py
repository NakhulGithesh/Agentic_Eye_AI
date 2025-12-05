#!/usr/bin/env python3
"""
Test script for scrcpy integration
"""

import os
import sys

# Add the Eye_code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Eye_code'))

try:
    from modules.mirroring.scrcpy_manager import ScrcpyManager
    
    print("Testing Scrcpy Integration...")
    
    # Get absolute path to scrcpy executable
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scrcpy_path = os.path.join(current_dir, 'Eye_code', 'platform-tools', 'scrcpy.exe')
    
    print(f"Scrcpy path: {scrcpy_path}")
    print(f"Scrcpy exists: {os.path.exists(scrcpy_path)}")
    
    if not os.path.exists(scrcpy_path):
        print("ERROR: Scrcpy executable not found!")
        sys.exit(1)
    
    # Configure scrcpy
    scrcpy_config = {
        'window_width': 350,
        'window_height': 600,
        'max_size': 800,
        'bit_rate': '8M',
        'max_fps': 30,
        'stay_awake': True,
        'show_touches': True,
        'disable_screensaver': True
    }
    
    # Initialize scrcpy manager
    manager = ScrcpyManager(scrcpy_path=scrcpy_path, config=scrcpy_config)
    print("SUCCESS: Scrcpy manager initialized")
    
    # Check device connection
    print("\nChecking device connection...")
    connected = manager.check_device_connected()
    devices = manager.get_connected_devices()
    
    print(f"Device connected: {connected}")
    print(f"Connected devices: {devices}")
    
    if connected:
        print("\nSUCCESS: Device found! You can now test mirroring in the main app.")
        print("   1. Run: python _main.py")
        print("   2. Click 'Show Phone' button")
        print("   3. Scrcpy window should open with your phone screen")
    else:
        print("\nWARNING: No device connected. To test mirroring:")
        print("   1. Connect your Android phone via USB")
        print("   2. Enable USB Debugging in Developer Options")
        print("   3. Allow computer access when prompted on phone")
        print("   4. Run this test again")
    
    # Get status
    status = manager.get_status()
    print(f"\nManager Status:")
    for key, value in status.items():
        print(f"   {key}: {value}")
        
except Exception as e:
    print(f"ERROR: Error testing scrcpy: {e}")
    import traceback
    traceback.print_exc()