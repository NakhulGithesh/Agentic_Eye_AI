"""
Control panels for Phone Mirror Application
Left and Right panel components with interactive buttons
"""

from PyQt5.QtWidgets import (QFrame, QVBoxLayout, QPushButton, QLabel, 
                            QHBoxLayout, QWidget, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette

from ._styles import COLORS


class ModernButton(QPushButton):
    """Custom button with modern styling and hover effects"""
    
    def __init__(self, text, icon=None, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.icon_text = icon or ""
        self.setMinimumHeight(50)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.apply_style()
        
    def apply_style(self):
        """Apply modern button styling"""
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['panel_bg']};
                border: 1px solid {COLORS['panel_border']};
                border-radius: 8px;
                color: {COLORS['text_primary']};
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
                font-weight: 500;
                padding: 8px 12px;
                text-align: left;
                margin: 2px;
            }}
            
            QPushButton:hover {{
                background-color: rgba(232, 241, 248, 220);
                border: 1px solid #A0C5D5;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 50);
            }}
            
            QPushButton:pressed {{
                background-color: rgba(209, 227, 240, 255);
                border: 1px solid #8FB5C5;
            }}
            
            QPushButton:disabled {{
                background-color: rgba(200, 200, 200, 100);
                color: #999999;
                border: 1px solid #CCCCCC;
            }}
        """)


class LeftControlPanel(QFrame):
    """Left control panel with device control buttons"""
    
    # Signals for button clicks
    light_clicked = pyqtSignal()
    speaker_clicked = pyqtSignal()
    wheelchair_clicked = pyqtSignal()
    close_menu_clicked = pyqtSignal()
    my_phone_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the left panel UI"""
        self.setFrameStyle(QFrame.StyledPanel)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 20, 15, 20)
        
        # Panel title
        title = QLabel("Device Controls")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"""
            font-size: 14px;
            font-weight: 600;
            color: {COLORS['text_primary']};
            margin-bottom: 10px;
            padding: 5px;
        """)
        layout.addWidget(title)
        
        # Control buttons
        self.light_btn = ModernButton("💡 Light", "💡")
        self.light_btn.clicked.connect(self.light_clicked.emit)
        
        self.speaker_btn = ModernButton("🔊 Speaker", "🔊")
        self.speaker_btn.clicked.connect(self.speaker_clicked.emit)
        
        self.wheelchair_btn = ModernButton("♿ Wheelchair", "♿")
        self.wheelchair_btn.clicked.connect(self.wheelchair_clicked.emit)
        
        self.close_menu_btn = ModernButton("❌ Close Menu", "❌")
        self.close_menu_btn.clicked.connect(self.close_menu_clicked.emit)
        
        self.my_phone_btn = ModernButton("📱 My Phone", "📱")
        self.my_phone_btn.clicked.connect(self.my_phone_clicked.emit)
        
        # Add buttons to layout
        layout.addWidget(self.light_btn)
        layout.addWidget(self.speaker_btn)
        layout.addWidget(self.wheelchair_btn)
        layout.addWidget(self.close_menu_btn)
        layout.addWidget(self.my_phone_btn)
        
        # Add stretch to push buttons to top
        layout.addStretch()
        
        # Status indicator
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(f"""
            font-size: 10px;
            color: {COLORS['text_secondary']};
            padding: 5px;
            background-color: rgba(200, 240, 200, 100);
            border-radius: 4px;
            margin-top: 10px;
        """)
        layout.addWidget(self.status_label)
    
    def update_status(self, status_text):
        """Update the status label"""
        self.status_label.setText(status_text)


class RightControlPanel(QFrame):
    """Right control panel with phone interaction buttons"""
    
    # Signals for button clicks
    volume_up_clicked = pyqtSignal()
    volume_down_clicked = pyqtSignal()
    scroll_up_clicked = pyqtSignal()
    scroll_down_clicked = pyqtSignal()
    interact_clicked = pyqtSignal()
    home_clicked = pyqtSignal()
    back_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the right panel UI"""
        self.setFrameStyle(QFrame.StyledPanel)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 20, 15, 20)
        
        # Panel title
        title = QLabel("Phone Controls")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"""
            font-size: 14px;
            font-weight: 600;
            color: {COLORS['text_primary']};
            margin-bottom: 10px;
            padding: 5px;
        """)
        layout.addWidget(title)
        
        # Volume controls
        volume_layout = QVBoxLayout()
        volume_label = QLabel("Volume")
        volume_label.setAlignment(Qt.AlignCenter)
        volume_label.setStyleSheet("font-size: 11px; font-weight: 500; margin: 5px;")
        volume_layout.addWidget(volume_label)
        
        self.volume_up_btn = ModernButton("🔊 Volume Up", "🔊")
        self.volume_up_btn.clicked.connect(self.volume_up_clicked.emit)
        volume_layout.addWidget(self.volume_up_btn)
        
        self.volume_down_btn = ModernButton("🔉 Volume Down", "🔉")
        self.volume_down_btn.clicked.connect(self.volume_down_clicked.emit)
        volume_layout.addWidget(self.volume_down_btn)
        
        layout.addLayout(volume_layout)
        
        # Scroll controls
        scroll_layout = QVBoxLayout()
        scroll_label = QLabel("Scroll")
        scroll_label.setAlignment(Qt.AlignCenter)
        scroll_label.setStyleSheet("font-size: 11px; font-weight: 500; margin: 5px;")
        scroll_layout.addWidget(scroll_label)
        
        self.scroll_up_btn = ModernButton("⬆️ Scroll Up", "⬆️")
        self.scroll_up_btn.clicked.connect(self.scroll_up_clicked.emit)
        scroll_layout.addWidget(self.scroll_up_btn)
        
        self.scroll_down_btn = ModernButton("⬇️ Scroll Down", "⬇️")
        self.scroll_down_btn.clicked.connect(self.scroll_down_clicked.emit)
        scroll_layout.addWidget(self.scroll_down_btn)
        
        layout.addLayout(scroll_layout)
        
        # Main interaction button
        self.interact_btn = ModernButton("👆 Interact", "👆")
        self.interact_btn.clicked.connect(self.interact_clicked.emit)
        self.interact_btn.setMinimumHeight(60)
        self.interact_btn.setStyleSheet(self.interact_btn.styleSheet() + """
            QPushButton {
                background-color: rgba(144, 202, 249, 180);
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: rgba(144, 202, 249, 220);
            }
        """)
        layout.addWidget(self.interact_btn)
        
        # Navigation controls
        nav_layout = QVBoxLayout()
        nav_label = QLabel("Navigation")
        nav_label.setAlignment(Qt.AlignCenter)
        nav_label.setStyleSheet("font-size: 11px; font-weight: 500; margin: 5px;")
        nav_layout.addWidget(nav_label)
        
        self.home_btn = ModernButton("🏠 Home Screen", "🏠")
        self.home_btn.clicked.connect(self.home_clicked.emit)
        nav_layout.addWidget(self.home_btn)
        
        self.back_btn = ModernButton("⬅️ Go Back", "⬅️")
        self.back_btn.clicked.connect(self.back_clicked.emit)
        nav_layout.addWidget(self.back_btn)
        
        layout.addLayout(nav_layout)
        
        # Add stretch to push buttons to top
        layout.addStretch()
        
        # Connection status
        self.connection_label = QLabel("Device: Disconnected")
        self.connection_label.setAlignment(Qt.AlignCenter)
        self.connection_label.setStyleSheet(f"""
            font-size: 10px;
            color: {COLORS['text_secondary']};
            padding: 5px;
            background-color: rgba(240, 200, 200, 100);
            border-radius: 4px;
            margin-top: 10px;
        """)
        layout.addWidget(self.connection_label)
    
    def update_connection_status(self, connected, device_count=0):
        """Update the connection status label"""
        if connected:
            self.connection_label.setText(f"Device: Connected ({device_count})")
            self.connection_label.setStyleSheet(f"""
                font-size: 10px;
                color: {COLORS['text_secondary']};
                padding: 5px;
                background-color: rgba(200, 240, 200, 100);
                border-radius: 4px;
                margin-top: 10px;
            """)
        else:
            self.connection_label.setText("Device: Disconnected")
            self.connection_label.setStyleSheet(f"""
                font-size: 10px;
                color: {COLORS['text_secondary']};
                padding: 5px;
                background-color: rgba(240, 200, 200, 100);
                border-radius: 4px;
                margin-top: 10px;
            """)


class CenterPhoneArea(QWidget):
    """Single unified widget for embedded phone screen display"""
    
    # Signals
    mirroring_toggled = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.embedded_window = None
        self.scrcpy_process = None
        self.is_mirroring = False
        self.init_ui()
        
    def init_ui(self):
        """Initialize the center area as a single container for embedded scrcpy"""
        # Remove frame styling - this is now a pure container
        self.setStyleSheet("""
            QWidget {
                background-color: #000000;
                border: 2px solid #C0D5E5;
                border-radius: 10px;
            }
        """)
        
        # Single layout with no margins for full embedding
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Container for embedded window
        self.embed_container = QWidget()
        self.embed_container.setStyleSheet("""
            QWidget {
                background-color: #000000;
                border: none;
            }
        """)
        self.embed_container.setMinimumSize(350, 600)
        
        self.layout.addWidget(self.embed_container)
        
        # Initially show a placeholder message
        self.show_placeholder()
    
    def show_placeholder(self):
        """Show placeholder content when not mirroring"""
        # Clear any existing content
        if self.embed_container.layout():
            for i in reversed(range(self.embed_container.layout().count())): 
                self.embed_container.layout().itemAt(i).widget().setParent(None)
        
        placeholder_layout = QVBoxLayout(self.embed_container)
        placeholder_layout.setAlignment(Qt.AlignCenter)
        
        placeholder_label = QLabel("Phone Screen\nWill Appear Here")
        placeholder_label.setAlignment(Qt.AlignCenter)
        placeholder_label.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-size: 18px;
                font-weight: bold;
                background-color: transparent;
                border: none;
            }
        """)
        
        placeholder_layout.addWidget(placeholder_label)
    
    def get_embed_container(self):
        """Get the container widget for embedding external windows"""
        return self.embed_container
    
    def set_mirroring_state(self, is_active):
        """Update the mirroring state"""
        self.is_mirroring = is_active
        if not is_active:
            self.show_placeholder()
    
    def get_embed_geometry(self):
        """Get the geometry for embedding the scrcpy window"""
        container_rect = self.embed_container.geometry()
        global_pos = self.embed_container.mapToGlobal(container_rect.topLeft())
        return {
            'x': global_pos.x(),
            'y': global_pos.y(), 
            'width': container_rect.width(),
            'height': container_rect.height()
        }