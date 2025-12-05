import subprocess
import os
import time
import logging
from typing import Optional, Dict, Any
import psutil

class ScrcpyManager:
    def __init__(self, scrcpy_path: str = None, config: Dict[str, Any] = None):
        self.scrcpy_path = scrcpy_path or os.path.join("platform-tools", "scrcpy.exe")
        self.adb_path = os.path.join("platform-tools", "adb.exe")
        self.process: Optional[subprocess.Popen] = None
        self.config = config or {}
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        if not os.path.exists(self.scrcpy_path):
            raise FileNotFoundError(f"scrcpy not found at {self.scrcpy_path}")
        if not os.path.exists(self.adb_path):
            raise FileNotFoundError(f"adb not found at {self.adb_path}")
    
    def check_device_connected(self) -> bool:
        try:
            result = subprocess.run([self.adb_path, "devices"], 
                                  capture_output=True, text=True, timeout=10)
            devices = result.stdout.strip().split('\n')[1:]
            connected_devices = [line for line in devices if line.strip() and '\tdevice' in line]
            
            self.logger.info(f"Found {len(connected_devices)} connected device(s)")
            return len(connected_devices) > 0
        except subprocess.TimeoutExpired:
            self.logger.error("ADB command timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error checking device connection: {e}")
            return False
    
    def get_connected_devices(self) -> list:
        try:
            result = subprocess.run([self.adb_path, "devices"], 
                                  capture_output=True, text=True, timeout=10)
            devices = result.stdout.strip().split('\n')[1:]
            connected_devices = []
            for line in devices:
                if line.strip() and '\tdevice' in line:
                    device_id = line.split('\t')[0]
                    connected_devices.append(device_id)
            return connected_devices
        except Exception as e:
            self.logger.error(f"Error getting connected devices: {e}")
            return []
    
    def is_scrcpy_running(self) -> bool:
        try:
            for process in psutil.process_iter(['pid', 'name']):
                if process.info['name'] == 'scrcpy.exe':
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error checking if scrcpy is running: {e}")
            return False
    
    def build_scrcpy_command(self) -> list:
        cmd = [self.scrcpy_path]
        
        cmd.extend(["--window-title", "AgenticEye Mirror"])
        
        if self.config.get("window_width") and self.config.get("window_height"):
            cmd.extend(["--window-width", str(self.config["window_width"])])
            cmd.extend(["--window-height", str(self.config["window_height"])])
        
        if self.config.get("max_size"):
            cmd.extend(["--max-size", str(self.config["max_size"])])
        
        if self.config.get("bit_rate"):
            cmd.extend(["--bit-rate", str(self.config["bit_rate"])])
        
        if self.config.get("max_fps"):
            cmd.extend(["--max-fps", str(self.config["max_fps"])])
        
        if self.config.get("stay_awake", True):
            cmd.append("--stay-awake")
        
        if self.config.get("turn_screen_off", False):
            cmd.append("--turn-screen-off")
        
        if self.config.get("show_touches", False):
            cmd.append("--show-touches")
        
        if self.config.get("disable_screensaver", True):
            cmd.append("--disable-screensaver")
        
        return cmd
    
    def start_mirroring(self) -> bool:
        if self.is_running():
            self.logger.warning("Mirroring is already running")
            return True
        
        if not self.check_device_connected():
            self.logger.error("No Android device connected")
            return False
        
        try:
            cmd = self.build_scrcpy_command()
            self.logger.info(f"Starting scrcpy with command: {' '.join(cmd)}")
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )
            
            time.sleep(2)
            
            if self.process.poll() is None:
                self.logger.info("Screen mirroring started successfully")
                return True
            else:
                stderr_output = self.process.stderr.read().decode() if self.process.stderr else "No error output"
                self.logger.error(f"Failed to start scrcpy. Error: {stderr_output}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting mirroring: {e}")
            return False
    
    def stop_mirroring(self) -> bool:
        if not self.is_running():
            self.logger.info("Mirroring is not running")
            return True
        
        try:
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.logger.warning("Process didn't terminate gracefully, forcing kill")
                    self.process.kill()
                    self.process.wait()
                
                self.process = None
            
            for process in psutil.process_iter(['pid', 'name']):
                if process.info['name'] == 'scrcpy.exe':
                    process.terminate()
                    self.logger.info(f"Terminated scrcpy process with PID {process.info['pid']}")
            
            self.logger.info("Screen mirroring stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping mirroring: {e}")
            return False
    
    def is_running(self) -> bool:
        if self.process and self.process.poll() is None:
            return True
        return self.is_scrcpy_running()
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running(),
            "device_connected": self.check_device_connected(),
            "connected_devices": self.get_connected_devices(),
            "scrcpy_path": self.scrcpy_path,
            "process_id": self.process.pid if self.process and self.process.poll() is None else None
        }
    
    def restart_mirroring(self) -> bool:
        self.logger.info("Restarting screen mirroring...")
        self.stop_mirroring()
        time.sleep(1)
        return self.start_mirroring()
    
    def demo(self) -> None:
        print("=== AgenticEye Mirroring Demo ===")
        print(f"Scrcpy path: {self.scrcpy_path}")
        print(f"ADB path: {self.adb_path}")
        
        status = self.get_status()
        print(f"Device connected: {status['device_connected']}")
        print(f"Connected devices: {status['connected_devices']}")
        print(f"Mirroring running: {status['is_running']}")
        
        if not status['device_connected']:
            print("Please connect your Android device via USB and enable USB debugging")
            return
        
        if status['is_running']:
            print("Stopping existing mirroring...")
            self.stop_mirroring()
        
        print("Starting screen mirroring...")
        if self.start_mirroring():
            print("Mirroring started successfully!")
            print("Press Enter to stop mirroring...")
            input()
            self.stop_mirroring()
            print("Mirroring stopped.")
        else:
            print("Failed to start mirroring.")

if __name__ == "__main__":
    manager = ScrcpyManager()
    manager.demo()