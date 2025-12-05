#!/usr/bin/env python3
"""
Phone Mirror Application Launcher
Main entry point for the PyQt5 application
"""

import sys
import os
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ui import MainWindow
except ImportError as e:
    print(f"Error importing UI components: {e}")
    print("Please ensure PyQt5 is installed: pip install PyQt5==5.15.10")
    sys.exit(1)


def main():
    """Main application entry point"""
    # Create the application
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Phone Mirror")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("PhoneMirror")
    
    # Enable high DPI scaling
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    try:
        # Create and show the main window
        window = MainWindow()
        window.show()
        
        # Start the event loop
        sys.exit(app.exec_())
        
    except Exception as e:
        # Show error dialog if something goes wrong
        error_msg = QMessageBox()
        error_msg.setIcon(QMessageBox.Critical)
        error_msg.setWindowTitle("Application Error")
        error_msg.setText(f"An error occurred while starting the application:\n\n{str(e)}")
        error_msg.exec_()
        sys.exit(1)


if __name__ == "__main__":
    main()