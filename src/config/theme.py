# filepath: /my-streamlit-app/my-streamlit-app/src/config/theme.py
THEME_SETTINGS = {
    "background_color": "#FFFFFF",  # White background
    "text_color": "#000000",        # Black text
    "accent_color": "#FF0000",      # Red accents
    "font_family": "Arial, sans-serif",
    "font_size": "16px",
}

def apply_theme():
    """Apply the defined theme settings to the Streamlit app."""
    import streamlit as st

    st.markdown(
        f"""
        <style>
        body {{
            background-color: {THEME_SETTINGS['background_color']};
            color: {THEME_SETTINGS['text_color']};
            font-family: {THEME_SETTINGS['font_family']};
            font-size: {THEME_SETTINGS['font_size']};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def set_theme():
    import streamlit as st

    st.set_page_config(
        page_title="CommentSense AI Analytics",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #dd1515;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dd1515;
    }
    .quality-high { color: #28a745; font-weight: bold; }
    .quality-medium { color: #ffc107; font-weight: bold; }
    .quality-low { color: #dc3545; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)