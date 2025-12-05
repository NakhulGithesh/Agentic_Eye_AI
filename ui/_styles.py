"""
Styling constants and color schemes for Phone Mirror Application
"""

# Color palette
COLORS = {
    'light_blue_grey': '#E8F1F8',
    'soft_grey_blue': '#D1E3F0',
    'panel_bg': 'rgba(232, 241, 248, 180)',  # Semi-transparent
    'panel_border': '#C0D5E5',
    'text_primary': '#2C3E50',
    'text_secondary': '#5D6D7E',
    'shadow': 'rgba(0, 0, 0, 30)'
}

# Main window styling
MAIN_WINDOW_STYLE = f"""
QMainWindow {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 {COLORS['light_blue_grey']},
                               stop:1 {COLORS['soft_grey_blue']});
}}
"""

# Panel styling
PANEL_STYLE = f"""
QFrame {{
    background-color: {COLORS['panel_bg']};
    border: 1px solid {COLORS['panel_border']};
    border-radius: 10px;
    margin: 10px;
}}

QFrame:hover {{
    background-color: rgba(232, 241, 248, 200);
}}
"""

# Label styling
LABEL_STYLE = f"""
QLabel {{
    color: {COLORS['text_primary']};
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 12px;
    font-weight: 500;
    background: transparent;
    padding: 5px;
}}

QLabel#title {{
    font-size: 16px;
    font-weight: 600;
    color: {COLORS['text_primary']};
}}

QLabel#subtitle {{
    font-size: 10px;
    color: {COLORS['text_secondary']};
}}
"""

# Combined stylesheet
STYLESHEET = f"""
{MAIN_WINDOW_STYLE}
{PANEL_STYLE}
{LABEL_STYLE}
"""

# Layout constants
LAYOUT = {
    'window_min_width': 1200,
    'window_min_height': 800,
    'panel_width': 200,
    'panel_margin': 10,
    'panel_spacing': 15,
    'content_margin': 20
}