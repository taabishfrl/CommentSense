import streamlit as st

def setup_page_and_css(LOGO_URI: str):
    st.set_page_config(
        page_title="CommentSense AI Analytics",
        page_icon="assets/mini.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(f"""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    /* Animated gradient background */
    .stApp {{
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c, #4facfe, #00f2fe);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        font-family: 'Poppins', sans-serif !important;
    }}
    @keyframes gradientBG {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    /* Floating particles */
    .stApp::before {{
        content: '';
        position: fixed;
        inset: 0;
        background-image:
          radial-gradient(circle at 20% 80%, rgba(255,255,255,.10) 1px, transparent 1px),
          radial-gradient(circle at 80% 20%, rgba(255,255,255,.15) 1px, transparent 1px),
          radial-gradient(circle at 40% 40%, rgba(255,255,255,.10) 1px, transparent 1px);
        background-size: 100px 100px, 150px 150px, 200px 200px;
        animation: float 20s ease-in-out infinite;
        pointer-events: none;
        z-index: -1;
    }}
    @keyframes float {{
        0%,100% {{ transform: translateY(0) rotate(0); }}
        50% {{ transform: translateY(-20px) rotate(180deg); }}
    }}

    /* Header styling — centered background logo */
    [data-testid="stHeader"] {{
      position: relative;
      background: white !important;
      backdrop-filter: blur(20px) !important;
      height: 90px;
    }}

    /* Fill the header and center the image */
    [data-testid="stHeader"]::before {{
      content: "";
      position: absolute;
      inset: 0;
      background-image: url("{LOGO_URI}");
      background-repeat: no-repeat;
      background-position: center center;
      background-size: auto 30px;
      z-index: 5;
      pointer-events: none;
    }}

    [data-testid="stSidebarContent"] {{ background: white; }}

    .block-container {{
      width: 80%;
      margin: 1.25rem auto !important;
      padding: 0 1rem !important;
      background: transparent !important;
      border: none !important;
      box-shadow: none !important;
      animation: slideIn 0.6s ease-out;
    }}
    @keyframes slideIn {{
      from {{ opacity: 0; transform: translateY(30px); }}
      to   {{ opacity: 1; transform: translateY(0);   }}
    }}

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
      color: white !important;
      font-weight: 600 !important;
    }}
    .stMarkdown h4 {{ color: black !important; font-weight: 600 !important; }}
    .stMarkdown h1 {{ font-size: 2.5rem !important; animation: titlePulse 2s ease-in-out infinite alternate; }}
    .stMarkdown p {{ color: rgba(255,255,255,0.9) !important; }}

    [data-testid="metric-container"] {{
      background: rgba(255,255,255,0.1) !important;
      backdrop-filter: blur(10px) !important;
      border: 1px solid rgba(255,255,255,0.2) !important;
      border-radius: 15px !important;
      padding: 1rem !important;
      transition: transform .3s ease, box-shadow .3s ease !important;
    }}
    [data-testid="metric-container"]:hover {{
      transform: translateY(-5px) !important;
      box-shadow: 0 8px 25px rgba(0,0,0,.2) !important;
    }}

    .stButton > button {{
      background: linear-gradient(45deg, #667eea, #764ba2) !important;
      color: white !important; border: none !important; border-radius: 25px !important;
      padding: .5rem 2rem !important; font-weight: 600 !important; transition: all .3s ease !important;
      box-shadow: 0 4px 15px rgba(102,126,234,.4) !important;
    }}
    .stButton > button:hover {{ transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(102,126,234,.6) !important; }}

    .stFileUploader > div {{
      background: rgba(255,255,255,0.1) !important; backdrop-filter: blur(10px) !important;
      border: 2px dashed rgba(255,255,255,.3) !important; border-radius: 15px !important;
    }}

    .js-plotly-plot .modebar,
    .js-plotly-plot .modebar-group {{
      background: transparent !important;
      box-shadow: none !important;
      border: none !important;
      padding: 0 !important;
      margin: 0 !important;
    }}

    .js-plotly-plot .modebar .modebar-btn {{
      background: #e0e0e0 !important;
      border: 1px solid #ccc !important;
      border-radius: 4px !important;
      padding: 2px !important;
      margin: 0 2px !important;
      align-items: center !important;
      justify-content: center !important;
      box-shadow: none !important;
      line-height: 0 !important;
    }}

    .js-plotly-plot .modebar .modebar-btn:hover,
    .js-plotly-plot .modebar .modebar-btn:focus {{
      background: #d0d0d0 !important;
      border-color: #bbb !important;
    }}

    .js-plotly-plot .modebar .modebar-btn svg {{
      fill: transparent !important;
    }}
    .js-plotly-plot .modebar .modebar-btn:hover svg {{
      fill: #000 !important;
    }}

    .stTabs [data-baseweb="tab-list"]{{
      gap: 8px;
      background: rgba(255,255,255,0.1) !important;
      backdrop-filter: blur(10px) !important;
      border-radius: 25px !important;
      padding: 6px !important;
    }}

    .stTabs [role="tab"]{{
      background: transparent !important;
      border-radius: 20px !important;
      padding: 8px 18px !important;
      color: #1b1b1b !important;
      transition: all .25s ease !important;
    }}

    .stTabs [role="tab"][aria-selected="true"]{{
      background: linear-gradient(45deg,#667eea,#764ba2) !important;
      box-shadow: 0 4px 15px rgba(102,126,234,.35) !important;
      color: #ffffff !important;
    }}
    .stTabs [role="tab"][aria-selected="true"] *{{
      color: #ffffff !important;
      fill: #ffffff !important;
    }}

    .stSelectbox > div > div, .stMultiSelect > div > div {{
      background: rgba(255,255,255,0.1) !important; backdrop-filter: blur(10px) !important;
      border: 1px solid rgba(255,255,255,0.2) !important; border-radius: 10px !important; color: white !important;
    }}
    .stDataFrame {{ border-radius: 15px !important; overflow: hidden !important; box-shadow: 0 8px 32px rgba(0,0,0,.1) !important; }}

    [data-testid="stChatInput"] {{
      background: rgba(255,255,255,0.1) !important; backdrop-filter: blur(15px) !important;
      border-radius: 25px !important; border: 1px solid rgba(255,255,255,0.2) !important;
    }}
    .stChatMessage {{
      background: rgba(255,255,255,0.1) !important; backdrop-filter: blur(10px) !important;
      border-radius: 15px !important; border: 1px solid rgba(255,255,255,0.2) !important; margin-bottom: 10px !important;
    }}
    .stAlert {{
      background: rgba(255,255,255,0.1) !important; backdrop-filter: blur(15px) !important;
      border-radius: 15px !important; border: 1px solid rgba(255,255,255,0.2) !important; color: white !important;
    }}

    hr {{ border: none !important; height: 1px !important;
         background: linear-gradient(90deg, transparent, rgba(255,255,255,.6), transparent) !important; margin: 2rem 0 !important; }}
    .js-plotly-plot .plotly .main-svg {{ border-radius: 15px !important; }}
    .stSuccess,.stError,.stWarning,.stInfo {{ border-radius: 15px !important; backdrop-filter: blur(10px) !important; border: 1px solid rgba(255,255,255,.2) !important; }}

    .cs-card-marker {{ display:none; }}

    [data-testid="stVerticalBlock"]:has(.cs-card-marker) {{
      position: relative;
      border-radius: 24px !important;
      padding: 24px !important;
      margin: 16px 0 !important;
      background: rgba(255,255,255,0.62) !important;
      -webkit-backdrop-filter: blur(10px);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255,255,255,0.35) !important;
      box-shadow: 0 18px 50px rgba(0,0,0,0.18) !important;
    }}

    [data-testid="stVerticalBlock"]:has(.cs-card-marker) :is(h1,h2,h3,h4,p,li,span,strong,em) {{ color: #1b1b1b !important; }}
    [data-testid="stVerticalBlock"]:has(.cs-card-marker) .js-plotly-plot .main-svg text {{ fill: #1b1b1b !important; }}
    [data-testid="stVerticalBlock"]:has(.cs-card-marker) [data-testid="metric-container"] {{
      background: rgba(255,255,255,0.55) !important; border: 1px solid rgba(0,0,0,0.06) !important;
    }}

    [data-testid="stVerticalBlock"]:has(.csv-btn-marker) button {{
      background: linear-gradient(45deg, #667eea, #764ba2) !important;
      color: #ffffff !important;
      border: none !important;
      border-radius: 26px !important;
      padding: 10px 20px !important;
      font-weight: 600 !important;
      box-shadow: 0 6px 18px rgba(102,126,234,0.35) !important;
      transition: transform .15s ease, box-shadow .15s ease, background .15s ease !important;
    }}
    [data-testid="stVerticalBlock"]:has(.csv-btn-marker) button:hover {{
      background: linear-gradient(45deg, #22c55e, #16a34a) !important;
      box-shadow: 0 8px 22px rgba(34,197,94,0.35) !important;
      transform: translateY(-1px) !important;
    }}
    [data-testid="stVerticalBlock"]:has(.csv-btn-marker) button:active {{
      transform: translateY(0) scale(0.98) !important;
      box-shadow: 0 3px 10px rgba(34,197,94,0.25) !important;
    }}
    [data-testid="stVerticalBlock"]:has(.csv-btn-marker) [data-testid="stDownloadButton"] button {{
      background: linear-gradient(45deg, #4f46e5, #7c3aed) !important;
      color: #fff !important;
      border-radius: 26px !important;
      border: none !important;
      font-weight: 600 !important;
    }}
    [data-testid="stVerticalBlock"]:has(.csv-btn-marker) [data-testid="stDownloadButton"] button:hover {{
      background: linear-gradient(45deg, #22c55e, #16a34a) !important;
      box-shadow: 0 8px 22px rgba(34,197,94,0.35) !important;
      transform: translateY(-1px) !important;
    }}
    [data-testid="stVerticalBlock"]:has(.csv-btn-marker) [data-testid="stDownloadButton"] button:active {{
      transform: translateY(0) scale(0.98) !important;
    }}

    .cs-video-title {{
      font-family: 'Poppins', sans-serif !important;
      font-weight: 600 !important;
      font-size: 1.1rem !important;
      color: #1b1b1b !important;
      margin: .25rem 0 .75rem 0 !important;
      text-align: center !important;
    }}
    </style>
    """, unsafe_allow_html=True)
