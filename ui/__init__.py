"""
UI package for Phone Mirror Application
"""

from ._main_window import MainWindow
from ._styles import STYLESHEET, LAYOUT, COLORS
from ._control_panels import LeftControlPanel, RightControlPanel, CenterPhoneArea

__all__ = ['MainWindow', 'STYLESHEET', 'LAYOUT', 'COLORS', 'LeftControlPanel', 'RightControlPanel', 'CenterPhoneArea']