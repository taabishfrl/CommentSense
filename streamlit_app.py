# CommentSense — Enhanced Streamlit Dashboard with Modern Styling
# -------------------------------------------------------------
# Run: streamlit run app.py

import base64
import pathlib
import re
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---- AI libs (all free) ----
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# =========================
# Utilities: logo in header
# =========================
def logo_data_uri(path="assets/logo.png") -> str:
    p = pathlib.Path(path)
    if not p.exists():
        st.warning(f"Logo not found at {p.resolve()}")
        return ""
    return "data:image/" + p.suffix[1:] + ";base64," + base64.b64encode(p.read_bytes()).decode()

LOGO_URI = logo_data_uri() 

# =========================
# Utilities: logo in header
# =========================
def logo_data_uri(path="assets/logo.png") -> str:
    p = pathlib.Path(path)
    if not p.exists():
        st.warning(f"Logo not found at {p.resolve()}")
        return ""
    return "data:image/" + p.suffix[1:] + ";base64," + base64.b64encode(p.read_bytes()).decode()

LOGO_URI = logo_data_uri()

# =========================
# Enhanced Page Configuration & Styling
# =========================
st.set_page_config(
    page_title="CommentSense AI Analytics",
    page_icon="assets/mini.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enhanced CSS with modern animations and glassmorphism
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
  height: 90px;                 /* tweak height as desired */
}}

/* Fill the header and center the image */
[data-testid="stHeader"]::before {{
  content: "";
  position: absolute;
  inset: 0;                      /* fill the header */
  background-image: url("{LOGO_URI}");  /* <-- single braces */
  background-repeat: no-repeat;
  background-position: center center;
  background-size: auto 30px; 
  z-index: 5;
  pointer-events: none;
}}

[data-testid="stSidebarContent"] {{ background: white; }}

/* ---- MAIN WRAPPER: make transparent, let CARDS provide glass look ---- */
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

/* Text styling (for dark background zones) */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
  color: white !important;
  font-weight: 600 !important;
}}
.stMarkdown h4 {{ color: black !important; font-weight: 600 !important; }}
.stMarkdown h1 {{ font-size: 2.5rem !important; animation: titlePulse 2s ease-in-out infinite alternate; }}
.stMarkdown p {{ color: rgba(255,255,255,0.9) !important; }}

/* Metrics */
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

/* Buttons, file uploader, tabs, selects... (unchanged from yours) */
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

/* ——— Plotly modebar: solid neutral buttons ——— */
.js-plotly-plot .modebar,
.js-plotly-plot .modebar-group {{
  background: transparent !important;
  box-shadow: none !important;
  border: none !important;
  padding: 0 !important;
  margin: 0 !important;
}}

.js-plotly-plot .modebar .modebar-btn {{
  background: #e0e0e0 !important;   /* light grey base */
  border: 1px solid #ccc !important;
  border-radius: 4px !important;
  padding: 2px !important;
  margin: 0 2px !important;
  align-items: center !important;
  justify-content: center !important;
  box-shadow: none !important;
  line-height: 0 !important;
}}

/* hover effect = darker grey */
.js-plotly-plot .modebar .modebar-btn:hover,
.js-plotly-plot .modebar .modebar-btn:focus {{
  background: #d0d0d0 !important;
  border-color: #bbb !important;
}}

/* icons = dark grey */
.js-plotly-plot .modebar .modebar-btn svg {{
  fill: transparent !important;
}}
.js-plotly-plot .modebar .modebar-btn:hover svg {{
  fill: #000 !important;
}}

/* Tab rail */
.stTabs [data-baseweb="tab-list"]{{
  gap: 8px;
  background: rgba(255,255,255,0.1) !important;
  backdrop-filter: blur(10px) !important;
  border-radius: 25px !important;
  padding: 6px !important;
}}

/* Unselected tab */
.stTabs [role="tab"]{{
  background: transparent !important;
  border-radius: 20px !important;
  padding: 8px 18px !important;
  color: #1b1b1b !important;        /* black when not selected */
  transition: all .25s ease !important;
}}

/* Selected tab */
.stTabs [role="tab"][aria-selected="true"]{{
  background: linear-gradient(45deg,#667eea,#764ba2) !important;
  box-shadow: 0 4px 15px rgba(102,126,234,.35) !important;
  color: #ffffff !important;        /* makes the tab element white */
}}

/* Make sure any nested label elements also turn white */
.stTabs [role="tab"][aria-selected="true"] *{{
  color: #ffffff !important;
  fill: #ffffff !important;         /* covers SVG icons if any */
}}

.stSelectbox > div > div, .stMultiSelect > div > div {{
  background: rgba(255,255,255,0.1) !important; backdrop-filter: blur(10px) !important;
  border: 1px solid rgba(255,255,255,0.2) !important; border-radius: 10px !important; color: white !important;
}}
.stDataFrame {{ border-radius: 15px !important; overflow: hidden !important; box-shadow: 0 8px 32px rgba(0,0,0,.1) !important; }}

/* Chat & alerts */
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

/* ===== Card containers (use with .cs-card-marker) ===== */
.cs-card-marker {{ display:none; }}

/* RELAXED selector: match any descendant marker, not just direct child */
[data-testid="stVerticalBlock"]:has(.cs-card-marker) {{
  position: relative;
  border-radius: 24px !important;
  padding: 24px !important;
  margin: 16px 0 !important;
  background: rgba(255,255,255,0.62) !important; /* translucent white */
  -webkit-backdrop-filter: blur(10px);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.35) !important;
  box-shadow: 0 18px 50px rgba(0,0,0,0.18) !important;
}}

/* Darker text inside cards for readability */
[data-testid="stVerticalBlock"]:has(.cs-card-marker) :is(h1,h2,h3,h4,p,li,span,strong,em) {{ color: #1b1b1b !important; }}

/* Keep Plotly/metrics readable on light cards */
[data-testid="stVerticalBlock"]:has(.cs-card-marker) .js-plotly-plot .main-svg text {{ fill: #1b1b1b !important; }}
[data-testid="stVerticalBlock"]:has(.cs-card-marker) [data-testid="metric-container"] {{
  background: rgba(255,255,255,0.55) !important; border: 1px solid rgba(0,0,0,0.06) !important;
}}

/* Scope to the Streamlit block that contains our csv-btn-marker */
[data-testid="stVerticalBlock"]:has(.csv-btn-marker) button {{
  /* base look */
  background: linear-gradient(45deg, #667eea, #764ba2) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 26px !important;
  padding: 10px 20px !important;
  font-weight: 600 !important;
  box-shadow: 0 6px 18px rgba(102,126,234,0.35) !important;
  transition: transform .15s ease, box-shadow .15s ease, background .15s ease !important;
}}

/* Hover = “save/ready” feel (green) */
[data-testid="stVerticalBlock"]:has(.csv-btn-marker) button:hover {{
  background: linear-gradient(45deg, #22c55e, #16a34a) !important; /* green */
  box-shadow: 0 8px 22px rgba(34,197,94,0.35) !important;
  transform: translateY(-1px) !important;
}}

/* Pressed/active */
[data-testid="stVerticalBlock"]:has(.csv-btn-marker) button:active {{
  transform: translateY(0) scale(0.98) !important;
  box-shadow: 0 3px 10px rgba(34,197,94,0.25) !important;
}}

/* Also style the download button (same block) */
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

/* Video title line */
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

# =========================
# Cached model loaders
# =========================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

@st.cache_resource
def load_zero_shot():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@st.cache_resource
def load_toxicity():
    return pipeline("text-classification", model="unitary/toxic-bert", truncation=True)

# =========================
# Data Loading & Auto-Detection Functions
# =========================
def detect_comment_column(df):
    """Automatically detect the comment/text column"""
    # Priority list of possible comment column names
    comment_priority = [
        'comment', 'text', 'textOriginal', 'content', 'message', 
        'body', 'review', 'feedback', 'comment_text', 'comment_body',
        'user_comment', 'comment_message', 'post_comment', 'comment_content',
        'review_text', 'feedback_text', 'commentary', 'response'
    ]
    
    # Check for exact matches first
    for col in comment_priority:
        if col in df.columns:
            return col
    
    # Check for partial matches (case insensitive)
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['comment', 'text', 'content', 'message', 'review', 'feedback', 'body']):
            return col
    
    # If no obvious match, return the first string column
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            return col
    
    # Last resort: return the first column
    return df.columns[0] if len(df.columns) > 0 else None

def detect_numeric_column(df, preferred_names):
    """Detect numeric columns with preferred names"""
    for col in preferred_names:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            return col
    
    # Find any numeric column
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    
    return None

def detect_date_column(df):
    """Detect date/time columns"""
    date_keywords = ['date', 'time', 'timestamp', 'created', 'published', 'posted', 'datetime']
    
    for col in date_keywords:
        if col in df.columns:
            return col
    
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in date_keywords):
            return col
    
    # Try to auto-detect datetime columns
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        try:
            # Try to convert to datetime
            pd.to_datetime(df[col].head(10))  # Test with first 10 rows
            return col
        except:
            continue
    
    return None

# =========================
# CommentAnalyzer
# =========================
class CommentAnalyzer:
    def __init__(self):
        self.embedder = load_embedder()
        self.sentiment_pipe = load_sentiment()
        self.zs_pipe = load_zero_shot()
        self.tox_pipe = load_toxicity()

        self.quality_keywords = {
            "high_quality": [
                "insightful","helpful","informative","detailed","thoughtful",
                "constructive","in-depth","compare","recommend","results","review",
            ],
            "spam_indicators": [
                "subscribe","check out","visit my","link in bio","click here",
                "follow me","dm me","whatsapp","promo","discount","coupon",
            ],
        }
        self.categories = ["skincare","fragrance","makeup","hair","other"]

    def analyze_sentiment(self, comment: str) -> str:
        try:
            out = self.sentiment_pipe(str(comment))[0]["label"].upper()
            return {"NEGATIVE":"Negative","NEUTRAL":"Neutral","POSITIVE":"Positive"}.get(out,"Neutral")
        except Exception:
            return "Neutral"

    def detect_toxicity(self, comment: str) -> float:
        try:
            out = self.tox_pipe(str(comment), return_all_scores=True)[0]
            for item in out:
                if item["label"].lower() == "toxic":
                    return float(item["score"])
            return 0.0
        except Exception:
            return 0.0

    def zero_shot_categories(self, comment: str):
        try:
            out = self.zs_pipe(str(comment), candidate_labels=self.categories, multi_label=True)
            labs = [l for l, s in zip(out["labels"], out["scores"]) if s >= 0.40]
            return labs or ["other"]
        except Exception:
            return ["other"]

    def relevance_to_post(self, comment: str, post_text: str) -> float:
        if not isinstance(post_text, str) or not post_text.strip():
            return 0.0
        emb = self.embedder.encode([str(comment), str(post_text)], normalize_embeddings=True)
        sim = float((emb[0] * emb[1]).sum())
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))

    def quality_score(self, comment: str, post_text: str) -> float:
        if not isinstance(comment, str) or not comment.strip():
            return 0.0
        c = comment.lower()
        rel = self.relevance_to_post(comment, post_text)
        tox = self.detect_toxicity(comment)
        sent = self.analyze_sentiment(comment)
        sent_s = {"Positive": 1.0, "Neutral": 0.5, "Negative": 0.0}.get(sent, 0.5)
        length_bonus = 0.15 if 30 <= len(c) <= 220 else (0.05 if len(c) > 220 else 0.0)
        kw_bonus = 0.15 if any(k in c for k in self.quality_keywords["high_quality"]) else 0.0
        spam_pen = 0.25 if any(s in c for s in self.quality_keywords["spam_indicators"]) else 0.0
        caps_pen = 0.15 if len(re.findall(r"[A-Z]", comment)) > max(1, len(comment)) * 0.35 else 0.0
        repeat_pen = 0.10 if re.search(r"(.)\\1{{4,}}", comment) else 0.0
        score = (0.45*rel + 0.25*sent_s + length_bonus + kw_bonus) - (0.35*tox + spam_pen + caps_pen + repeat_pen)
        return float(max(0.0, min(1.0, score)))

# =========================
# Sample data
# =========================
def load_sample_data():
    sample_comments = [
        "This tutorial was incredibly helpful! The step-by-step breakdown made it easy to follow.",
        "Great video! Could you do one about advanced techniques?",
        "First! Love your content!",
        "Really detailed explanation. Thank you!",
        "Subscribe to my channel for more content like this!",
        "I disagree with your approach, but I appreciate the thorough analysis.",
        "Wow!!! So good!!!",
        "Very informative. The examples were particularly useful.",
        "Can you do skincare routine for sensitive skin?",
        "Check out my latest video! Link in bio!",
        "This fragrance sounds interesting. What are the main notes?",
        "Your makeup tutorial inspired me to try new looks.",
        "The gaming review was spot on. Have you tried the new update?",
        "Thanks for the tech review. Very comprehensive.",
        "aaaaaaaaawesome video!!!!",
    ]
    video_ids = ["VID001","VID002","VID003"] * 5
    titles = {"VID001":"Hydrating Serum 101","VID002":"Summer Fragrance Picks","VID003":"Everyday Makeup Tips"}
    df = pd.DataFrame({
        "video_id": video_ids[:len(sample_comments)],
        "comment": sample_comments,
        "likes": np.random.randint(0, 50, len(sample_comments)),
        "shares": np.random.randint(0, 10, len(sample_comments)),
        "saves": np.random.randint(0, 10, len(sample_comments)),
        "timestamp": pd.date_range("2024-01-01", periods=len(sample_comments), freq="H"),
    })
    df["title"] = df["video_id"].map(titles)
    return df

# =========================
# Plot helpers
# =========================
def style_plotly_chart(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        margin=dict(t=50, r=20, b=40, l=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(font=dict(color="white")),
    )
    fig.update_xaxes(color="white", gridcolor="rgba(255,255,255,0.2)")
    fig.update_yaxes(color="white", gridcolor="rgba(255,255,255,0.2)")
    if fig.layout.annotations:
        for a in fig.layout.annotations:
            a.font = a.font or dict()
            a.font.color = "white"
    return fig

# =========================
# Main App
# =========================
def main():
    analyzer = CommentAnalyzer()

    # ---------- Sidebar: data load ----------
    st.sidebar.markdown("#### Data Management")
    uploaded_file = st.sidebar.file_uploader(
        "Upload comments CSV/XLSX", type=["csv","xlsx"], help="Upload your comment data for analysis"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            
            # Auto-detect columns for universal compatibility
            comment_col = detect_comment_column(df)
            like_col = detect_numeric_column(df, ['likes', 'likeCount', 'favorites', 'engagement', 'score', 'upvotes'])
            date_col = detect_date_column(df)
            id_col = None
            
            # Try to find an ID column
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['id', 'video', 'post', 'item', 'product']):
                    id_col = col
                    break
            
            if id_col is None and len(df.columns) > 1:
                id_col = df.columns[0]  # Use first column as ID if none found
            
            # Rename columns for internal consistency
            column_mapping = {}
            if comment_col:
                df = df.rename(columns={comment_col: 'comment'})
                column_mapping[comment_col] = 'comment'
                st.sidebar.success(f"✅ Using '{comment_col}' as comment column")
            
            if like_col:
                df = df.rename(columns={like_col: 'likes'})
                column_mapping[like_col] = 'likes'
                st.sidebar.success(f"✅ Using '{like_col}' as likes column")
            else:
                df['likes'] = 0  # Default value
                st.sidebar.info("ℹ️ No likes column found. Using default value (0).")
            
            if date_col:
                df = df.rename(columns={date_col: 'timestamp'})
                column_mapping[date_col] = 'timestamp'
                # Convert to datetime if it's a string
                if df['timestamp'].dtype == 'object':
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    except:
                        df['timestamp'] = pd.Timestamp.now()
                st.sidebar.success(f"✅ Using '{date_col}' as timestamp column")
            else:
                df['timestamp'] = pd.Timestamp.now()  # Default value
                st.sidebar.info("ℹ️ No timestamp column found. Using current time.")
            
            if id_col:
                df = df.rename(columns={id_col: 'video_id'})
                column_mapping[id_col] = 'video_id'
                st.sidebar.success(f"✅ Using '{id_col}' as video ID column")
            else:
                df['video_id'] = 'Item_' + df.index.astype(str)  # Generate IDs
                st.sidebar.info("ℹ️ No ID column found. Generating automatic IDs.")
            
            # Show data preview
            st.sidebar.subheader("📋 Data Preview")
            st.sidebar.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
            
            if st.sidebar.checkbox("Show first 3 rows"):
                st.sidebar.dataframe(df.head(3))
                
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.info("Falling back to sample data for demonstration.")
            df = load_sample_data()
    else:
        st.info("📝 No file uploaded. Using sample data for demonstration.")
        df = load_sample_data()

    # Let user manually override column mapping if needed
    st.sidebar.subheader("🛠️ Manual Column Mapping (Optional)")
    
    all_columns = df.columns.tolist()
    current_comment_col = 'comment' if 'comment' in df.columns else all_columns[0] if all_columns else None
    
    if all_columns:
        manual_comment_col = st.sidebar.selectbox(
            "Select comment/text column", 
            options=all_columns,
            index=all_columns.index(current_comment_col) if current_comment_col in all_columns else 0
        )
        
        # Update if user selected a different column
        if manual_comment_col != current_comment_col and manual_comment_col != 'comment':
            df = df.rename(columns={manual_comment_col: 'comment'})
            st.sidebar.info(f"Using '{manual_comment_col}' as comment column")

    # Ensure we have the required comment column
    if 'comment' not in df.columns:
        st.error("Could not identify a text column in your data.")
        st.info("Please ensure your CSV contains at least one column with text content.")
        return

    # ---------- Pick/prepare post_text column ----------
    post_text_col = None

    for c in ["caption", "title", "video_caption", "post_text", "text", "description"]:

        if c in df.columns:
            post_text_col = c
            break
    if post_text_col is None:
        if "video_id" in df.columns:
            df[post_text_col := "post_text"] = df["video_id"].astype(str)
        else:
            df[post_text_col := "post_text"] = ""

    # ---------- AI Analysis with progress (in a card) ----------
    card_progress = st.empty()  # use st.empty so we can fully clear later
    with card_progress.container():
        st.markdown('<div class="cs-card-marker"></div>', unsafe_allow_html=True)

        st.markdown("### AI Analysis in Progress…")
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Analyzing sentiment...")
        df["sentiment"] = df["comment"].apply(analyzer.analyze_sentiment)
        progress_bar.progress(25)

        status_text.text("Detecting toxicity...")
        df["toxicity"] = df["comment"].apply(analyzer.detect_toxicity)
        progress_bar.progress(50)

        status_text.text("Calculating relevance...")
        df["relevance"] = df.apply(lambda r: analyzer.relevance_to_post(r["comment"], r[post_text_col]), axis=1)
        progress_bar.progress(75)

        status_text.text("Categorizing content...")
        def cats_or_other(row):
            return analyzer.zero_shot_categories(row["comment"]) if row["relevance"] >= 0.40 else ["other"]
        df["categories"] = df.apply(cats_or_other, axis=1)

        df["quality_score"] = df.apply(lambda r: analyzer.quality_score(r["comment"], r[post_text_col]), axis=1)
        progress_bar.progress(100)

        status_text.text("✅ Analysis complete!")

    # fully clear the entire card (so it disappears)
    card_progress.empty()

    # ---------- Labels ----------
    df["quality_category"] = pd.cut(df["quality_score"], bins=[-1,0.4,0.7,1.0], labels=["Low","Medium","High"])
    df["is_spam"] = (df["quality_score"] < 0.25) | (df["toxicity"] > 0.6)

    # ---------- Filters ----------
    st.sidebar.markdown("#### Filters")
    if "video_id" in df.columns:
        vids = st.sidebar.multiselect("📹 Video IDs", df["video_id"].unique().tolist(),
                                      default=df["video_id"].unique().tolist())
        df = df[df["video_id"].isin(vids)]

    quality_sel = st.sidebar.multiselect("⭐ Quality Level", ["High","Medium","Low"],
                                         default=["High","Medium","Low"])
    df_filtered = df[df["quality_category"].isin(quality_sel)]

    # ---- Video title display (from filtered data) ----
    if "title" in df_filtered.columns and not df_filtered["title"].dropna().empty:
        # If multiple videos are selected, show the most common title + a count hint
        unique_titles = df_filtered["title"].dropna().astype(str).unique().tolist()
        if len(unique_titles) == 1:
            title_to_show = unique_titles[0]
            subtitle = f"🎬 {title_to_show}"
        else:
            mode_title = df_filtered["title"].mode()[0]
            subtitle = f"🎬 {mode_title} (+{len(unique_titles)-1} more)"
    else:
        subtitle = "🎬 (unknown)"

    st.markdown(f'<div class="cs-video-title">{subtitle}</div>', unsafe_allow_html=True)

    # ------------------ BLOCK 1: Title + KPIs (card) ------------------
    card1 = st.container()
    with card1:
        st.markdown('<div class="cs-card-marker"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; padding: 0 0 1rem 0;">
            <h1 style="font-size: 2.4rem; margin-bottom: 0.35rem;">CommentSense AI Analytics</h1>
            <p style="font-size: 1.05rem; opacity: 0.9;">
                AI-powered analysis of <strong>relevance, sentiment, toxicity, spam</strong> and a
                <strong>quality-weighted SoE</strong> metric.
            </p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("💬 Total Comments", len(df_filtered), delta=None)
        with c2:
            qcr = (df_filtered["quality_category"] == "High").mean() * 100 if len(df_filtered) else 0
            st.metric("🎯 QCR (Quality %)", f"{qcr:.1f}%", delta=f"{qcr-20:.1f}%" if qcr > 20 else None)
        with c3:
            spam = (df_filtered["is_spam"]).mean() * 100 if len(df_filtered) else 0
            spam_delta = f"-{spam:.1f}%" if spam < 30 else f"+{spam:.1f}%"
            st.metric("🚫 Spam %", f"{spam:.1f}%", delta=spam_delta, delta_color="inverse")
        with c4:
            avg_quality = df_filtered["quality_score"].mean() if len(df_filtered) else 0
            st.metric("⚡ Avg Quality Score", f"{avg_quality:.2f}",
                      delta=f"{avg_quality-0.5:.2f}" if avg_quality > 0.5 else None)
    # ------------------ END BLOCK 1 ------------------

    # ------------------ BLOCK 2: Analytics + Table + Export + Insights (card) ------------------
    card2 = st.container()
    with card2:
        st.markdown('<div class="cs-card-marker"></div>', unsafe_allow_html=True)

        st.markdown("### Analytics Dashboard")
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Quality","😊 Sentiment","🏷️ Categories","🚫 Spam"])

        with tab1:
            colA, colB = st.columns(2)
            with colA:
                vc = df_filtered["quality_category"].value_counts()
                fig = px.pie(values=vc.values, names=vc.index, title="Quality Distribution")
                st.plotly_chart(style_plotly_chart(fig), use_container_width=True)
            with colB:
                if "video_id" in df_filtered.columns:
                    g = df_filtered.groupby(["video_id","quality_category"]).size().unstack(fill_value=0)
                    fig2 = px.bar(g, title="Quality by Video")
                    st.plotly_chart(style_plotly_chart(fig2), use_container_width=True)

        with tab2:
            colA, colB = st.columns(2)
            with colA:
                sc = df_filtered["sentiment"].value_counts()
                fig = px.bar(x=sc.index, y=sc.values, title="Sentiment Distribution")
                st.plotly_chart(style_plotly_chart(fig), use_container_width=True)
            with colB:
                if len(df_filtered):
                    fig = px.box(df_filtered, x="sentiment", y="quality_score", title="Quality Score by Sentiment")
                    st.plotly_chart(style_plotly_chart(fig), use_container_width=True)

        with tab3:
            all_cats = []
            for L in df_filtered["categories"]:
                all_cats.extend(L)
            if all_cats:
                counts = Counter(all_cats)
                fig = px.bar(x=list(counts.keys()), y=list(counts.values()), title="Category Distribution")
                st.plotly_chart(style_plotly_chart(fig), use_container_width=True)
            else:
                st.info("🔍 No categories detected in current filter.")

        with tab4:
            colA, colB = st.columns(2)
            with colA:
                spam_counts = df_filtered["is_spam"].value_counts()
                names = ["✅ Clean" if not k else "🚫 Spam" for k in spam_counts.index]
                fig = px.pie(values=spam_counts.values, names=names, title="Spam vs Clean Distribution")
                st.plotly_chart(style_plotly_chart(fig), use_container_width=True)
            with colB:
                g = df_filtered.groupby(["is_spam","quality_category"]).size().unstack(fill_value=0)
                fig = px.bar(g, title="Quality Distribution by Spam Status")
                st.plotly_chart(style_plotly_chart(fig), use_container_width=True)

        # Detailed Comment Analysis
        st.markdown("### Detailed Comment Analysis")
        with st.expander("View Detailed Data Table", expanded=False):
            show_cols = ["comment","quality_score","quality_category","sentiment","toxicity","relevance","categories","is_spam"]
            missing = [c for c in show_cols if c not in df_filtered.columns]
            for m in missing:
                df_filtered[m] = np.nan
            styled_df = df_filtered[show_cols].reset_index(drop=True)
            st.dataframe(
                styled_df, use_container_width=True, height=500,
                column_config={
                    "quality_score": st.column_config.ProgressColumn("Quality Score", min_value=0, max_value=1, format="%.2f"),
                    "toxicity":      st.column_config.ProgressColumn("Toxicity",      min_value=0, max_value=1, format="%.2f"),
                    "relevance":     st.column_config.ProgressColumn("Relevance",     min_value=0, max_value=1, format="%.2f"),
                },
            )

        # Export Results
        st.markdown("### Export Results")
        colDL, colInfo = st.columns([1,2])
        with colDL:
            # --- marker so our CSS can target ONLY this area ---
            st.markdown('<span class="csv-btn-marker"></span>', unsafe_allow_html=True)

            if st.button("Prepare CSV Download", type="primary", key="prep_csv"):
                st.session_state._csv = df.to_csv(index=False)
                st.success("CSV prepared successfully!")

            if "_csv" in st.session_state:
                st.download_button(
                    "Download Analysis CSV",
                    data=st.session_state._csv,
                    file_name=f"comment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    type="secondary",
                    key="dl_csv",
                )
        with colInfo:
            st.info(" **Export includes:** All original data plus AI analysis results (sentiment, toxicity, relevance, categories, quality scores)")

        # Insights (inside same card)
        st.markdown("### AI-Generated Insights & Recommendations")
        insights = []
        if len(df_filtered):
            if qcr > 30:
                insights.append("🎉 **High QCR Detected** — Your content is sparking meaningful, on-topic discussion!")
            elif qcr < 15:
                insights.append("📈 **QCR Improvement Opportunity** — Try clearer CTAs or more specific captions to prompt constructive replies.")
            if spam > 20:
                insights.append("🚨 **Elevated Spam Levels** — Consider moderation rules or blocking promotional keywords.")
            pos_sentiment = (df_filtered["sentiment"] == "Positive").mean() * 100
            if pos_sentiment > 60:
                insights.append("😊 **Positive Audience Sentiment** — Great job! Consider scaling this content theme.")
            elif pos_sentiment < 30:
                insights.append("😐 **Neutral/Negative Sentiment** — Consider adjusting content approach to increase engagement.")
            high_toxicity = (df_filtered["toxicity"] > 0.5).mean() * 100
            if high_toxicity > 10:
                insights.append("⚠️ **Toxicity Alert** — Consider implementing stricter comment moderation.")
            avg_relevance = df_filtered["relevance"].mean()
            if avg_relevance > 0.7:
                insights.append("🎯 **High Relevance Score** — Comments are well-aligned with your content topics.")
            elif avg_relevance < 0.4:
                insights.append("🔄 **Low Relevance Score** — Comments may be off-topic. Consider more focused content themes.")
        if insights:
            for tip in insights:
                if any(k in tip for k in ["🎉","😊","🎯"]):
                    st.success(tip)
                elif any(k in tip for k in ["⚠️","🚨"]):
                    st.error(tip)
                else:
                    st.info(tip)
        else:
            st.info("🔍 Upload your data to get personalized insights!")
    # ------------------ END BLOCK 2 ------------------

    # ------------------ BLOCK 3: Chat (card) ------------------
    card3 = st.container()
    with card3:
        st.markdown('<div class="cs-card-marker"></div>', unsafe_allow_html=True)
        st.markdown("### 🤖 Chat with CommentSense AI")
        st.markdown("*Ask me about your data, QCR metrics, or get specific examples!*")

        if "messages" not in st.session_state:
            st.session_state.messages = [{
                "role":"assistant",
                "content":"👋 Hi! I'm your CommentSense AI assistant. Ask me things like:\n\n• **\"What is QCR?\"**\n• **\"Show top posts\"**\n• **\"Find skincare positive examples\"**\n• **\"Why is spam high?\"**\n\nWhat would you like to explore?",
            }]

        for m in st.session_state.messages:
            with st.chat_message(m["role"], avatar="🤖" if m["role"]=="assistant" else "👤"):
                st.markdown(m["content"])

        def chat_reply(prompt: str) -> str:
            p = prompt.lower().strip()

            if "qcr" in p or ("quality" in p and ("ratio" in p or "comment" in p)):
                cur_qcr = (df_filtered["quality_category"] == "High").mean() * 100 if len(df_filtered) else 0
                return (
                    f"🎯 **Quality Comment Ratio (QCR)** is the percentage of high-quality comments in your dataset.\n\n"
                    f"📊 **Current QCR: {cur_qcr:.1f}%**\n\n"
                    "🔍 **How we calculate quality:**\n• Relevance to your content (using AI embeddings)\n• Positive sentiment analysis\n"
                    "• Content length and specificity\n• Absence of spam/toxic language\n\n"
                    "💡 **Good QCR benchmarks:**\n• 30%+ = Excellent engagement\n• 15-30% = Good engagement\n• <15% = Room for improvement"
                )

            if "top" in p and ("post" in p or "video" in p):
                df_temp = df_filtered.copy()
                cols = [c for c in ["likes","shares","saves"] if c in df_temp.columns]
                if cols:
                    std_vals = df_temp[cols].std(ddof=0).replace(0,1)
                    z = (df_temp[cols] - df_temp[cols].mean()) / std_vals
                    df_temp["_soe"] = z.mean(axis=1)
                else:
                    df_temp["_soe"] = 0.0
                df_temp["_qsoe"] = df_temp["_soe"] * df_temp["quality_score"]
                if "video_id" in df_temp.columns:
                    top = df_temp.groupby("video_id")["_qsoe"].mean().sort_values(ascending=False).head(5)
                    if len(top) > 0:
                        st.dataframe(
                            top.rename("Quality-weighted SoE").reset_index(),
                            use_container_width=True,
                            column_config={
                                "Quality-weighted SoE": st.column_config.ProgressColumn(
                                    "Q-SoE Score",
                                    min_value=float(top.min()) if len(top) > 0 else 0,
                                    max_value=float(top.max()) if len(top) > 0 else 1,
                                    format="%.3f",
                                )
                            },
                        )
                        return ("🏆 **Top posts by Quality-weighted Share of Engagement (Q-SoE)** shown above!\n\n"
                                "Q-SoE combines traditional engagement metrics with our AI quality assessment to identify content that generates both "
                                "high engagement AND meaningful discussion.")
                    else:
                        return "📊 No video data available for ranking."
                return "🤖 I computed Q-SoE, but couldn't find video groupings in your data."

            if "find" in p or "example" in p:
                want_cat = None
                for c in ["skincare","fragrance","makeup","hair","other"]:
                    if c in p:
                        want_cat = c; break
                want_sent = None
                for s in ["positive","neutral","negative"]:
                    if s in p:
                        want_sent = s.capitalize(); break

                q = df_filtered.copy()
                if want_cat:
                    q = q[q["categories"].apply(lambda L: want_cat in L if isinstance(L, list) else False)]
                if want_sent:
                    q = q[q["sentiment"] == want_sent]
                q = q.sort_values("quality_score", ascending=False).head(5)

                if len(q) > 0:
                    display_df = q[["comment","quality_score","sentiment","categories"]].reset_index(drop=True)
                    st.dataframe(
                        display_df, use_container_width=True,
                        column_config={
                            "comment": st.column_config.TextColumn("Comment", width="large"),
                            "quality_score": st.column_config.ProgressColumn("Quality Score", min_value=0, max_value=1, format="%.2f"),
                        },
                    )
                    filter_text = f" in **{want_cat}** category" if want_cat else ""
                    filter_text += f" with **{want_sent}** sentiment" if want_sent else ""
                    return (f"🔍 Found **{len(q)}** high-quality examples{filter_text}!\n\n"
                            "These comments score highest on our quality metrics, combining relevance, constructiveness, and engagement value.")
                else:
                    return "🤷‍♂️ No examples found matching your criteria. Try adjusting the category or sentiment filter."

            if "spam" in p and ("why" in p or "high" in p or "cause" in p):
                spam_rate = (df_filtered["is_spam"]).mean() * 100 if len(df_filtered) else 0
                spam_comments = df_filtered[df_filtered["is_spam"]]
                spam_examples = spam_comments["comment"].head(3).tolist() if len(spam_comments) > 0 else []
                spam_details = (
                    f"**Spam Analysis: {spam_rate:.1f}%** of your comments are flagged as spam.\n\n"
                    "🔍 **Common spam indicators we detect:**\n"
                    "• Promotional keywords (\"subscribe\", \"check out\", \"link in bio\")\n"
                    "• Very short or repetitive content\n"
                    "• High toxicity scores\n"
                    "• Off-topic comments with low relevance\n\n"
                    "💡 **Reduce spam by:**\n"
                    "• Setting up keyword filters\n"
                    "• Enabling comment moderation\n"
                    "• Using community guidelines prompts\n"
                    "• Encouraging specific, on-topic questions"
                )
                if spam_examples:
                    spam_details += "\n\n **Sample spam comments:**\n" + "\n".join([f'• \"{c[:80]}...\"' for c in spam_examples])
                return spam_details

            if "help" in p or "what can you do" in p:
                return (
                    "🤖 **I'm your CommentSense AI assistant!** Here's what I can help with:\n\n"
                    "📊 **Analytics Explanations:**\n• Explain QCR, Q-SoE, and other metrics\n• Break down quality scoring methodology\n\n"
                    "🏆 **Performance Insights:**\n• Identify your top-performing content\n• Analyze engagement patterns\n• Compare video performance\n\n"
                    "🔍 **Content Discovery:**\n• Find high-quality comment examples\n• Filter by category (skincare, fragrance, etc.)\n• Sort by sentiment (positive, neutral, negative)\n\n"
                    "🚨 **Problem Diagnosis:**\n• Explain spam patterns\n• Identify engagement issues\n• Suggest improvement strategies\n\n"
                    "💬 **Just ask me naturally!**"
                )

            return (
                "🤔 I'm not sure about that. Try:\n\n"
                "🎯 **\"What is QCR?\"**\n"
                "🏆 **\"Show top posts\"**\n"
                "🔍 **\"Find [category] [sentiment] examples\"**\n"
                "🚨 **\"Why is spam high?\"**"
            )

        if user_prompt := st.chat_input("Ask about QCR, top posts, examples, or anything else..."):
            st.session_state.messages.append({"role":"user","content":user_prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_prompt)
            with st.spinner("🤖 Analyzing..."):
                reply = chat_reply(user_prompt)
            st.session_state.messages.append({"role":"assistant","content":reply})
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(reply)
    # ------------------ END BLOCK 3 ------------------

    # ---------- Footer ----------
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem; opacity: .7;">
            <p><strong>CommentSense AI Analytics</strong> | Powered by Transformer Models & Advanced NLP</p>
            <p>Built with ❤️ using Streamlit, HuggingFace Transformers, and Sentence Transformers</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---- Enhanced run
if __name__ == "__main__":
    main()