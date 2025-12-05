import os
from typing import Dict, Any

class MirroringSettings:
    def __init__(self):
        self.default_config = {
            "window_width": 800,
            "window_height": 600,
            "max_size": 1024,
            "bit_rate": "8M",
            "max_fps": 60,
            "stay_awake": True,
            "turn_screen_off": False,
            "show_touches": False,
            "disable_screensaver": True,
            "auto_start": False,
            "log_level": "INFO"
        }
        
        self.scrcpy_config = {
            "scrcpy_path": os.path.join("platform-tools", "scrcpy.exe"),
            "adb_path": os.path.join("platform-tools", "adb.exe")
        }
    
    def get_mirroring_config(self) -> Dict[str, Any]:
        return self.default_config.copy()
    
    def get_scrcpy_paths(self) -> Dict[str, str]:
        return self.scrcpy_config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        self.default_config.update(new_config)

class AgenticEyeSettings:
    def __init__(self):
        self.mirroring = MirroringSettings()
        
        self.project_config = {
            "project_name": "AgenticEye BCI",
            "version": "1.0.0",
            "modules": {
                "mirroring": True,
                "eye_tracking": False,
                "eeg": False
            }
        }
    
    def get_all_settings(self) -> Dict[str, Any]:
        return {
            "project": self.project_config,
            "mirroring": self.mirroring.get_mirroring_config(),
            "paths": self.mirroring.get_scrcpy_paths()
        }

SETTINGS = AgenticEyeSettings()