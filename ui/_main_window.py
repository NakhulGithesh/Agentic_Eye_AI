"""
Main window class for Phone Mirror Application
"""

import sys
import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                            QFrame, QLabel, QApplication, QMessageBox)
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt5.QtGui import QPalette, QFont

from ._styles import STYLESHEET, LAYOUT, COLORS
from ._control_panels import LeftControlPanel, RightControlPanel, CenterPhoneArea

# Add the Eye_code directory to path for scrcpy manager
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
eye_code_path = os.path.join(project_root, 'Eye_code')
sys.path.insert(0, eye_code_path)
from modules.mirroring.scrcpy_manager import ScrcpyManager


class MainWindow(QMainWindow):
    """Main application window with three-panel layout and scrcpy integration"""
    
    def __init__(self):
        super().__init__()
        self.scrcpy_manager = None
        self.init_scrcpy()
        self.init_ui()
        self.setup_timers()
        
    def init_scrcpy(self):
        """Initialize the scrcpy manager"""
        try:
            # Configure scrcpy with appropriate settings for embedding
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
            
            # Get absolute paths to scrcpy and adb executables  
            platform_tools_dir = os.path.join(project_root, 'Eye_code', 'platform-tools')
            scrcpy_path = os.path.join(platform_tools_dir, 'scrcpy.exe')
            adb_path = os.path.join(platform_tools_dir, 'adb.exe')
            
            print(f"🔧 PROJECT ROOT: {project_root}")
            print(f"🔧 PLATFORM TOOLS DIR: {platform_tools_dir}")
            print(f"🔧 Looking for scrcpy at: {scrcpy_path}")
            print(f"🔧 Scrcpy exists: {os.path.exists(scrcpy_path)}")
            print(f"🔧 Looking for adb at: {adb_path}")
            print(f"🔧 ADB exists: {os.path.exists(adb_path)}")
            
            if not os.path.exists(scrcpy_path):
                print(f"❌ CRITICAL: Scrcpy not found at {scrcpy_path}")
            if not os.path.exists(adb_path):
                print(f"❌ CRITICAL: ADB not found at {adb_path}")
            
            # Test ADB directly
            if os.path.exists(adb_path):
                try:
                    import subprocess
                    test_result = subprocess.run([adb_path, "version"], 
                                               capture_output=True, text=True, timeout=5)
                    print(f"🔧 ADB version test - Return code: {test_result.returncode}")
                    print(f"🔧 ADB version output: {test_result.stdout}")
                    if test_result.stderr:
                        print(f"🔧 ADB version stderr: {test_result.stderr}")
                except Exception as e:
                    print(f"❌ Failed to test ADB: {e}")
            
            # Create a custom ScrcpyManager class that uses absolute paths
            class FixedScrcpyManager(ScrcpyManager):
                def __init__(self, scrcpy_path, config, adb_path):
                    # Initialize parent without path validation
                    import logging
                    import psutil
                    self.scrcpy_path = scrcpy_path
                    self.adb_path = adb_path
                    self.process = None
                    self.config = config or {}
                    
                    self.logger = logging.getLogger(__name__)
                    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
                    
                    # Check if files exist with absolute paths
                    if not os.path.exists(self.scrcpy_path):
                        raise FileNotFoundError(f"scrcpy not found at {self.scrcpy_path}")
                    if not os.path.exists(self.adb_path):
                        raise FileNotFoundError(f"adb not found at {self.adb_path}")
                
                def is_scrcpy_running(self):
                    """Check if any scrcpy process is running"""
                    try:
                        import psutil
                        for process in psutil.process_iter(['pid', 'name']):
                            if process.info['name'] and 'scrcpy' in process.info['name'].lower():
                                return True
                        return False
                    except Exception as e:
                        self.logger.error(f"Error checking if scrcpy is running: {e}")
                        return False
                
                def is_running(self):
                    """Check if mirroring is running"""
                    # Check both our process and system-wide scrcpy processes
                    if self.process and self.process.poll() is None:
                        return True
                    return self.is_scrcpy_running()
            
            # Create the fixed scrcpy manager
            self.scrcpy_manager = FixedScrcpyManager(scrcpy_path, scrcpy_config, adb_path)
            print("Scrcpy manager initialized successfully")
            
        except Exception as e:
            print(f"Warning: Could not initialize scrcpy manager: {e}")
            self.scrcpy_manager = None
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Phone Mirror - Control Interface")
        self.setMinimumSize(LAYOUT['window_min_width'], LAYOUT['window_min_height'])
        
        # Set the main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main horizontal layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create the three panels with enhanced controls
        self.left_panel = LeftControlPanel()
        self.center_area = CenterPhoneArea()
        self.right_panel = RightControlPanel()
        
        # Set fixed widths for side panels
        self.left_panel.setFixedWidth(LAYOUT['panel_width'])
        self.right_panel.setFixedWidth(LAYOUT['panel_width'])
        
        # Add panels to main layout
        main_layout.addWidget(self.left_panel)
        main_layout.addWidget(self.center_area, 1)  # Stretch factor of 1
        main_layout.addWidget(self.right_panel)
        
        # Connect panel signals
        self.connect_panel_signals()
        
        # Apply stylesheet
        self.setStyleSheet(STYLESHEET)
        
        # Center window on screen
        self.center_window()
        
    def connect_panel_signals(self):
        """Connect signals from control panels"""
        # Left panel signals
        self.left_panel.light_clicked.connect(self.on_light_clicked)
        self.left_panel.speaker_clicked.connect(self.on_speaker_clicked)
        self.left_panel.wheelchair_clicked.connect(self.on_wheelchair_clicked)
        self.left_panel.close_menu_clicked.connect(self.on_close_menu_clicked)
        self.left_panel.my_phone_clicked.connect(self.on_my_phone_clicked)
        
        # Right panel signals
        self.right_panel.volume_up_clicked.connect(self.on_volume_up_clicked)
        self.right_panel.volume_down_clicked.connect(self.on_volume_down_clicked)
        self.right_panel.scroll_up_clicked.connect(self.on_scroll_up_clicked)
        self.right_panel.scroll_down_clicked.connect(self.on_scroll_down_clicked)
        self.right_panel.interact_clicked.connect(self.on_interact_clicked)
        self.right_panel.home_clicked.connect(self.on_home_clicked)
        self.right_panel.back_clicked.connect(self.on_back_clicked)
        
        # Center area signals - updated for new implementation
        # Note: Center area no longer has toggle buttons - controlled from side panels
        
    def setup_timers(self):
        """Setup timers for status updates"""
        # Timer to check device connection status
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_device_status)
        self.status_timer.start(2000)  # Check every 2 seconds
        print("Status timer started - checking device every 2 seconds", flush=True)
        
        # Do initial status update
        print("Performing initial device status update...", flush=True)
        self.update_device_status()
    
    # Button handler methods
    def on_light_clicked(self):
        """Handle light button click"""
        self.left_panel.update_status("Light control activated")
        QMessageBox.information(self, "Light Control", "Light control feature activated!")
    
    def on_speaker_clicked(self):
        """Handle speaker button click"""
        self.left_panel.update_status("Speaker control activated")
        QMessageBox.information(self, "Speaker Control", "Speaker control feature activated!")
    
    def on_wheelchair_clicked(self):
        """Handle wheelchair button click"""
        self.left_panel.update_status("Wheelchair control activated")
        QMessageBox.information(self, "Wheelchair Control", "Wheelchair control feature activated!")
    
    def on_close_menu_clicked(self):
        """Handle close menu button click"""
        self.close()
    
    def on_my_phone_clicked(self):
        """Handle my phone button click - toggle scrcpy mirroring"""
        if self.scrcpy_manager:
            self.toggle_mirroring()
        else:
            QMessageBox.warning(self, "Scrcpy Error", "Scrcpy manager not available!")
    
    def on_volume_up_clicked(self):
        """Handle volume up button click"""
        if self.scrcpy_manager and self.scrcpy_manager.check_device_connected():
            # Send volume up command via adb
            self.send_adb_command("input keyevent KEYCODE_VOLUME_UP")
        else:
            QMessageBox.warning(self, "Device Error", "No device connected!")
    
    def on_volume_down_clicked(self):
        """Handle volume down button click"""
        if self.scrcpy_manager and self.scrcpy_manager.check_device_connected():
            # Send volume down command via adb
            self.send_adb_command("input keyevent KEYCODE_VOLUME_DOWN")
        else:
            QMessageBox.warning(self, "Device Error", "No device connected!")
    
    def on_scroll_up_clicked(self):
        """Handle scroll up button click"""
        if self.scrcpy_manager and self.scrcpy_manager.check_device_connected():
            # Send scroll up command (swipe down to scroll up)
            self.send_adb_command("input swipe 500 800 500 400 300")
        else:
            QMessageBox.warning(self, "Device Error", "No device connected!")
    
    def on_scroll_down_clicked(self):
        """Handle scroll down button click"""
        if self.scrcpy_manager and self.scrcpy_manager.check_device_connected():
            # Send scroll down command (swipe up to scroll down)
            self.send_adb_command("input swipe 500 400 500 800 300")
        else:
            QMessageBox.warning(self, "Device Error", "No device connected!")
    
    def on_interact_clicked(self):
        """Handle interact button click"""
        if self.scrcpy_manager and self.scrcpy_manager.check_device_connected():
            # Send tap command at center of screen
            self.send_adb_command("input tap 500 1000")
        else:
            QMessageBox.warning(self, "Device Error", "No device connected!")
    
    def on_home_clicked(self):
        """Handle home button click"""
        if self.scrcpy_manager and self.scrcpy_manager.check_device_connected():
            # Send home key command
            self.send_adb_command("input keyevent KEYCODE_HOME")
        else:
            QMessageBox.warning(self, "Device Error", "No device connected!")
    
    def on_back_clicked(self):
        """Handle back button click"""
        if self.scrcpy_manager and self.scrcpy_manager.check_device_connected():
            # Send back key command
            self.send_adb_command("input keyevent KEYCODE_BACK")
        else:
            QMessageBox.warning(self, "Device Error", "No device connected!")
    
    def toggle_mirroring(self):
        """Toggle mirroring state - called from phone button"""
        if self.scrcpy_manager and self.scrcpy_manager.is_running():
            self.stop_mirroring()
        else:
            self.start_mirroring()
    
    def send_adb_command(self, command):
        """Send ADB command to connected device"""
        if self.scrcpy_manager:
            try:
                import subprocess
                adb_path = self.scrcpy_manager.adb_path
                full_command = [adb_path, "shell"] + command.split()
                subprocess.run(full_command, capture_output=True, text=True, timeout=5)
            except Exception as e:
                print(f"Error sending ADB command: {e}")
    
    def start_mirroring(self):
        """Start phone mirroring"""
        if not self.scrcpy_manager:
            QMessageBox.warning(self, "Scrcpy Error", "Scrcpy manager not available!")
            return
        
        # Check if device is connected first
        if not self.scrcpy_manager.check_device_connected():
            QMessageBox.warning(self, "Device Error", 
                              "No Android device connected!\n\n"
                              "Please:\n"
                              "1. Connect your phone via USB\n"
                              "2. Enable USB Debugging\n"
                              "3. Allow computer access on phone")
            return
        
        # Update UI to show starting
        self.left_panel.update_status("Starting mirroring...")
        
        # Start mirroring in a separate thread to avoid UI blocking
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, self._do_start_mirroring)
    
    def _do_start_mirroring(self):
        """Actually start the mirroring process with embedding"""
        try:
            print(f"🔄 Starting mirroring process in UI...")
            
            # Get the geometry for embedding the scrcpy window
            embed_geometry = self.center_area.get_embed_geometry()
            print(f"📐 Embedding geometry: {embed_geometry}")
            
            # Double-check device connection before attempting
            if not self.scrcpy_manager.check_device_connected():
                self.left_panel.update_status("No device connected")
                QMessageBox.warning(self, "Device Error", 
                                  "Device connection lost!\n\n"
                                  "Please ensure:\n"
                                  "• Phone is still connected via USB\n"
                                  "• USB Debugging is still enabled\n"
                                  "• Try reconnecting the USB cable")
                return
            
            print(f"🔄 Device confirmed connected, starting scrcpy...")
            
            if self.scrcpy_manager.start_mirroring(embed_geometry):
                self.center_area.set_mirroring_state(True)
                self.left_panel.update_status("Mirroring active")
                print("✅ UI: Scrcpy mirroring started successfully with embedding!")
            else:
                self.left_panel.update_status("Mirroring failed")
                print("❌ UI: Scrcpy failed to start, showing detailed error...")
                
                # Get more detailed error information
                error_details = "Failed to start phone mirroring!\n\n"
                error_details += "Debug steps:\n"
                error_details += "1. Check console output for detailed errors\n"
                error_details += "2. Verify USB Debugging is enabled\n"
                error_details += "3. Try 'adb devices' in command prompt\n"
                error_details += "4. Restart ADB server if needed\n\n"
                error_details += "Common fixes:\n"
                error_details += "• Disconnect and reconnect USB cable\n"
                error_details += "• Change USB connection mode to File Transfer\n"
                error_details += "• Accept USB debugging authorization on phone"
                
                QMessageBox.warning(self, "Mirroring Error", error_details)
        except Exception as e:
            error_msg = f"Error starting mirroring: {str(e)}"
            print(f"❌ UI Exception: {error_msg}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Mirroring Error", 
                               f"Critical error during mirroring start:\n\n{error_msg}\n\n"
                               f"Check console for full stack trace.")
    
    def stop_mirroring(self):
        """Stop phone mirroring"""
        if not self.scrcpy_manager:
            return
            
        try:
            self.left_panel.update_status("Stopping mirroring...")
            
            if self.scrcpy_manager.stop_mirroring():
                self.center_area.set_mirroring_state(False)
                self.left_panel.update_status("Mirroring stopped")
                print("✅ Scrcpy mirroring stopped successfully!")
            else:
                self.left_panel.update_status("Stop failed")
                print("❌ Failed to stop scrcpy mirroring")
        except Exception as e:
            error_msg = f"Error stopping mirroring: {str(e)}"
            print(f"❌ {error_msg}")
    
    def update_device_status(self):
        """Update device connection status"""
        if self.scrcpy_manager:
            try:
                connected = self.scrcpy_manager.check_device_connected()
                devices = self.scrcpy_manager.get_connected_devices()
                
                # Debug output with immediate flush
                print(f"Device Status Update:", flush=True)
                print(f"   Connected: {connected}", flush=True)
                print(f"   Devices: {devices}", flush=True)
                print(f"   Device count: {len(devices)}", flush=True)
                print(f"   Mirroring running: {self.scrcpy_manager.is_running()}", flush=True)
                
                # Update right panel
                self.right_panel.update_connection_status(connected, len(devices))
                
                # Update status based on connection and mirroring state
                if connected and not self.scrcpy_manager.is_running():
                    self.left_panel.update_status("Device connected")
                elif connected and self.scrcpy_manager.is_running():
                    self.left_panel.update_status("Mirroring active")
                    self.center_area.set_mirroring_state(True)
                elif not connected:
                    self.left_panel.update_status("No device")
                    self.center_area.set_mirroring_state(False)
                    
            except Exception as e:
                print(f"ERROR: Error updating device status: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("WARNING: No scrcpy manager available")
            self.right_panel.update_connection_status(False, 0)
        
    
    def center_window(self):
        """Center the window on the screen"""
        if QApplication.desktop() is not None:
            screen = QApplication.desktop().screenGeometry()
            size = self.geometry()
            self.move(
                (screen.width() - size.width()) // 2,
                (screen.height() - size.height()) // 2
            )
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop mirroring if running
        if self.scrcpy_manager and self.scrcpy_manager.is_running():
            self.scrcpy_manager.stop_mirroring()
        
        # Stop status timer
        if hasattr(self, 'status_timer'):
            self.status_timer.stop()
        
        event.accept()