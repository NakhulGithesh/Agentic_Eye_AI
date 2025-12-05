#!/usr/bin/env python3
"""
Debug script for scrcpy manager issues
"""

import os
import sys

print("=== SCRCPY DEBUG ===")

# Check current directory
print(f"Current directory: {os.getcwd()}")

# Check file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {current_dir}")

project_root = current_dir  # The current directory IS the project root
print(f"Project root: {project_root}")

platform_tools_dir = os.path.join(project_root, 'Eye_code', 'platform-tools')
print(f"Platform tools directory: {platform_tools_dir}")
print(f"Platform tools exists: {os.path.exists(platform_tools_dir)}")

scrcpy_path = os.path.join(platform_tools_dir, 'scrcpy.exe')
adb_path = os.path.join(platform_tools_dir, 'adb.exe')

print(f"Scrcpy path: {scrcpy_path}")
print(f"Scrcpy exists: {os.path.exists(scrcpy_path)}")
print(f"ADB path: {adb_path}")
print(f"ADB exists: {os.path.exists(adb_path)}")

# Try to add Eye_code to path
eye_code_path = os.path.join(project_root, 'Eye_code')
print(f"Eye_code path: {eye_code_path}")
print(f"Eye_code exists: {os.path.exists(eye_code_path)}")

if eye_code_path not in sys.path:
    sys.path.insert(0, eye_code_path)
    print("Added Eye_code to sys.path")

# Check if modules directory exists
modules_path = os.path.join(eye_code_path, 'modules')
print(f"Modules path: {modules_path}")
print(f"Modules exists: {os.path.exists(modules_path)}")

mirroring_path = os.path.join(modules_path, 'mirroring')
print(f"Mirroring path: {mirroring_path}")  
print(f"Mirroring exists: {os.path.exists(mirroring_path)}")

scrcpy_manager_path = os.path.join(mirroring_path, 'scrcpy_manager.py')
print(f"Scrcpy manager path: {scrcpy_manager_path}")
print(f"Scrcpy manager exists: {os.path.exists(scrcpy_manager_path)}")

# Try importing the scrcpy manager
try:
    from modules.mirroring.scrcpy_manager import ScrcpyManager
    print("SUCCESS: Imported ScrcpyManager")
    
    # Try creating instance
    manager = ScrcpyManager(scrcpy_path=scrcpy_path)
    print("SUCCESS: Created ScrcpyManager instance")
    
    # Override adb path
    manager.adb_path = adb_path
    print("SUCCESS: Set ADB path")
    
    # Test device check
    connected = manager.check_device_connected()
    print(f"Device check result: {connected}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("=== END DEBUG ===")