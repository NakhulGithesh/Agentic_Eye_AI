#!/usr/bin/env python3
import sys
import os
import argparse
import logging
from typing import Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.mirroring.scrcpy_manager import ScrcpyManager
from config.settings import SETTINGS

class AgenticEyeLauncher:
    def __init__(self):
        self.settings = SETTINGS
        self.mirroring_manager: Optional[ScrcpyManager] = None
        
        logging.basicConfig(
            level=getattr(logging, self.settings.mirroring.default_config.get("log_level", "INFO")),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_mirroring(self) -> bool:
        try:
            paths = self.settings.mirroring.get_scrcpy_paths()
            config = self.settings.mirroring.get_mirroring_config()
            
            self.mirroring_manager = ScrcpyManager(
                scrcpy_path=paths["scrcpy_path"],
                config=config
            )
            
            self.logger.info("Mirroring module initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize mirroring: {e}")
            return False
    
    def start_mirroring(self) -> bool:
        if not self.mirroring_manager:
            if not self.initialize_mirroring():
                return False
        
        return self.mirroring_manager.start_mirroring()
    
    def stop_mirroring(self) -> bool:
        if not self.mirroring_manager:
            self.logger.info("Mirroring manager not initialized")
            return True
        
        return self.mirroring_manager.stop_mirroring()
    
    def get_status(self) -> dict:
        status = {
            "project": self.settings.project_config,
            "mirroring": None
        }
        
        if self.mirroring_manager:
            status["mirroring"] = self.mirroring_manager.get_status()
        else:
            status["mirroring"] = {"initialized": False}
        
        return status
    
    def interactive_mode(self):
        print("=== AgenticEye BCI Launcher ===")
        print("1. Start mirroring")
        print("2. Stop mirroring")
        print("3. Check status")
        print("4. Demo mirroring")
        print("5. Exit")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-5): ").strip()
                
                if choice == "1":
                    print("Starting mirroring...")
                    if self.start_mirroring():
                        print("✓ Mirroring started successfully!")
                    else:
                        print("✗ Failed to start mirroring")
                
                elif choice == "2":
                    print("Stopping mirroring...")
                    if self.stop_mirroring():
                        print("✓ Mirroring stopped successfully!")
                    else:
                        print("✗ Failed to stop mirroring")
                
                elif choice == "3":
                    status = self.get_status()
                    print("\n=== Status ===")
                    print(f"Project: {status['project']['project_name']} v{status['project']['version']}")
                    
                    if status['mirroring'].get('initialized', True):
                        mirroring = status['mirroring']
                        print(f"Device connected: {mirroring.get('device_connected', 'Unknown')}")
                        print(f"Connected devices: {mirroring.get('connected_devices', [])}")
                        print(f"Mirroring active: {mirroring.get('is_running', False)}")
                        if mirroring.get('process_id'):
                            print(f"Process ID: {mirroring['process_id']}")
                    else:
                        print("Mirroring: Not initialized")
                
                elif choice == "4":
                    if not self.mirroring_manager:
                        self.initialize_mirroring()
                    if self.mirroring_manager:
                        self.mirroring_manager.demo()
                    else:
                        print("Failed to initialize mirroring for demo")
                
                elif choice == "5":
                    print("Exiting...")
                    self.stop_mirroring()
                    break
                
                else:
                    print("Invalid choice. Please enter 1-5.")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                self.stop_mirroring()
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="AgenticEye BCI Launcher")
    parser.add_argument("--start-mirroring", action="store_true", help="Start screen mirroring")
    parser.add_argument("--stop-mirroring", action="store_true", help="Stop screen mirroring")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--demo", action="store_true", help="Run mirroring demo")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    launcher = AgenticEyeLauncher()
    
    if args.start_mirroring:
        success = launcher.start_mirroring()
        sys.exit(0 if success else 1)
    
    elif args.stop_mirroring:
        success = launcher.stop_mirroring()
        sys.exit(0 if success else 1)
    
    elif args.status:
        status = launcher.get_status()
        print(f"Project: {status['project']['project_name']} v{status['project']['version']}")
        if status['mirroring'].get('initialized', True):
            mirroring = status['mirroring']
            print(f"Device connected: {mirroring.get('device_connected', 'Unknown')}")
            print(f"Mirroring active: {mirroring.get('is_running', False)}")
        sys.exit(0)
    
    elif args.demo:
        if launcher.initialize_mirroring():
            launcher.mirroring_manager.demo()
        sys.exit(0)
    
    else:
        launcher.interactive_mode()

if __name__ == "__main__":
    main()