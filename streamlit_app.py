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
# Enhanced Page Configuration & Styling
# =========================
st.set_page_config(
    page_title="CommentSense AI Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enhanced CSS with modern animations and glassmorphism
st.markdown(f"""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Animated gradient background */
.stApp {{
    background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c, #4facfe, #00f2fe);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    font-family: 'Inter', sans-serif;
}}

@keyframes gradientBG {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}

/* Floating particles animation */
.stApp::before {{
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        radial-gradient(circle at 20% 80%, rgba(255, 255, 255, 0.1) 1px, transparent 1px),
        radial-gradient(circle at 80% 20%, rgba(255, 255, 255, 0.15) 1px, transparent 1px),
        radial-gradient(circle at 40% 40%, rgba(255, 255, 255, 0.1) 1px, transparent 1px);
    background-size: 100px 100px, 150px 150px, 200px 200px;
    animation: float 20s ease-in-out infinite;
    pointer-events: none;
    z-index: -1;
}}

@keyframes float {{
    0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
    50% {{ transform: translateY(-20px) rotate(180deg); }}
}}

/* Header styling */
[data-testid="stHeader"] {{
    position: relative;
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(20px) !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.2) !important;
    height: 72px;
}}

[data-testid="stHeader"]::before {{
    content: "";
    position: absolute;
    top: 50%;
    left: 50%;
    width: 200px;
    height: 56px;
    transform: translate(-50%, -50%);
    background-image: url("{LOGO_URI}");
    background-repeat: no-repeat;
    background-position: center center;
    background-size: contain;
    z-index: 5;
    pointer-events: none;
    filter: drop-shadow(0 0 10px rgba(255, 255, 255, 0.5));
    animation: logoGlow 3s ease-in-out infinite alternate;
}}

@keyframes logoGlow {{
    from {{ filter: drop-shadow(0 0 10px rgba(255, 255, 255, 0.5)); }}
    to {{ filter: drop-shadow(0 0 20px rgba(255, 255, 255, 0.8)); }}
}}

/* Glassmorphism sidebar */
.css-1d391kg {{
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 0 20px 20px 0 !important;
}}

/* Main content with glassmorphism */
.block-container {{
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(15px) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    padding: 2rem !important;
    margin: 1rem !important;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37) !important;
    animation: slideIn 0.6s ease-out;
}}

@keyframes slideIn {{
    from {{ 
        opacity: 0; 
        transform: translateY(30px); 
    }}
    to {{ 
        opacity: 1; 
        transform: translateY(0); 
    }}
}}

/* Enhanced text styling */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
    color: white !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important;
    font-weight: 600 !important;
}}

.stMarkdown h1 {{
    font-size: 2.5rem !important;
    animation: titlePulse 2s ease-in-out infinite alternate;
}}

@keyframes titlePulse {{
    from {{ text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
    to {{ text-shadow: 2px 2px 20px rgba(255,255,255,0.6); }}
}}

.stMarkdown p {{
    color: rgba(255, 255, 255, 0.9) !important;
    font-weight: 400 !important;
}}

/* Enhanced metrics */
[data-testid="metric-container"] {{
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 15px !important;
    padding: 1rem !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1) !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
}}

[data-testid="metric-container"]:hover {{
    transform: translateY(-5px) !important;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2) !important;
}}

/* Enhanced buttons */
.stButton > button {{
    background: linear-gradient(45deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 0.5rem 2rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}}

.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}}

/* Enhanced file uploader */
.stFileUploader > div {{
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(10px) !important;
    border: 2px dashed rgba(255, 255, 255, 0.3) !important;
    border-radius: 15px !important;
    transition: all 0.3s ease !important;
}}

.stFileUploader > div:hover {{
    border-color: rgba(255, 255, 255, 0.6) !important;
    background: rgba(255, 255, 255, 0.15) !important;
}}

/* Enhanced tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 2px;
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 25px !important;
    padding: 4px !important;
}}

.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    color: white !important;
    border-radius: 20px !important;
    padding: 8px 16px !important;
    transition: all 0.3s ease !important;
}}

.stTabs [aria-selected="true"] {{
    background: linear-gradient(45deg, #667eea, #764ba2) !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}}

/* Enhanced selectbox and multiselect */
.stSelectbox > div > div {{
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 10px !important;
    color: white !important;
}}

.stMultiSelect > div > div {{
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 10px !important;
}}

/* Enhanced dataframe */
.stDataFrame {{
    border-radius: 15px !important;
    overflow: hidden !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
}}

/* Chat styling */
[data-testid="stChatInput"] {{
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(15px) !important;
    border-radius: 25px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}}

.stChatMessage {{
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 15px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    margin-bottom: 10px !important;
}}

/* Enhanced info/warning boxes */
.stAlert {{
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(15px) !important;
    border-radius: 15px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: white !important;
}}

/* Sidebar enhancements */
.sidebar .sidebar-content {{
    background: rgba(255, 255, 255, 0.05) !important;
}}

.sidebar .element-container {{
    animation: fadeInLeft 0.6s ease-out;
}}

@keyframes fadeInLeft {{
    from {{
        opacity: 0;
        transform: translateX(-30px);
    }}
    to {{
        opacity: 1;
        transform: translateX(0);
    }}
}}

/* Loading spinner enhancement */
.stSpinner > div {{
    border-top-color: #667eea !important;
    filter: drop-shadow(0 0 10px #667eea) !important;
}}

/* Checkbox styling */
.stCheckbox > label {{
    color: white !important;
    font-weight: 500 !important;
}}

/* Enhanced divider */
hr {{
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.6), transparent) !important;
    margin: 2rem 0 !important;
}}

/* Plotly chart enhancements */
.js-plotly-plot .plotly .main-svg {{
    border-radius: 15px !important;
}}

/* Success/error message styling */
.stSuccess, .stError, .stWarning, .stInfo {{
    border-radius: 15px !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}}

</style>
""", unsafe_allow_html=True)

# =========================
# Cached model loaders (unchanged)
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
# CommentAnalyzer class (unchanged)
# =========================
class CommentAnalyzer:
    def __init__(self):
        self.embedder = load_embedder()
        self.sentiment_pipe = load_sentiment()
        self.zs_pipe = load_zero_shot()
        self.tox_pipe = load_toxicity()

        self.quality_keywords = {
            "high_quality": [
                "insightful", "helpful", "informative", "detailed", "thoughtful",
                "constructive", "in-depth", "compare", "recommend", "results", "review",
            ],
            "spam_indicators": [
                "subscribe", "check out", "visit my", "link in bio", "click here",
                "follow me", "dm me", "whatsapp", "promo", "discount", "coupon",
            ],
        }
        self.categories = ["skincare", "fragrance", "makeup", "hair", "other"]

    def analyze_sentiment(self, comment: str) -> str:
        try:
            out = self.sentiment_pipe(str(comment))[0]["label"].upper()
            return {"NEGATIVE": "Negative", "NEUTRAL": "Neutral", "POSITIVE": "Positive"}.get(out, "Neutral")
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
        repeat_pen = 0.10 if re.search(r"(.)\1{4,}", comment) else 0.0

        score = (0.45 * rel + 0.25 * sent_s + length_bonus + kw_bonus) - (0.35 * tox + spam_pen + caps_pen + repeat_pen)
        return float(max(0.0, min(1.0, score)))

# =========================
# Sample data (unchanged)
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
    video_ids = ["VID001", "VID002", "VID003"] * 5
    titles = {
        "VID001": "Hydrating Serum 101",
        "VID002": "Summer Fragrance Picks",
        "VID003": "Everyday Makeup Tips",
    }
    df = pd.DataFrame(
        {
            "video_id": video_ids[: len(sample_comments)],
            "comment": sample_comments,
            "likes": np.random.randint(0, 50, len(sample_comments)),
            "shares": np.random.randint(0, 10, len(sample_comments)),
            "saves": np.random.randint(0, 10, len(sample_comments)),
            "timestamp": pd.date_range("2024-01-01", periods=len(sample_comments), freq="H"),
        }
    )
    df["title"] = df["video_id"].map(titles)
    return df

# =========================
# Enhanced chart functions
# =========================

def style_plotly_chart(fig: go.Figure) -> go.Figure:
    # Shared layout tweaks for dark glass UI
    fig.update_layout(
        margin=dict(t=50, r=20, b=40, l=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(font=dict(color="white"))
    )
    fig.update_xaxes(color="white", gridcolor="rgba(255,255,255,0.2)")
    fig.update_yaxes(color="white", gridcolor="rgba(255,255,255,0.2)")
    # Make sure any annotation text is readable
    if fig.layout.annotations:
        for a in fig.layout.annotations:
            a.font = a.font or dict()
            a.font.color = "white"
    return fig

def create_enhanced_pie_chart(values, names, title, colors=None):
    """Create an enhanced pie chart with better styling"""
    if colors is None:
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
    
    fig = go.Figure(data=[go.Pie(
        labels=names, 
        values=values,
        hole=0.4,
        textfont_size=12,
        marker=dict(colors=colors, line=dict(color='rgba(255,255,255,0.8)', width=2))
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(color='white', size=16)),
        showlegend=True,
        legend=dict(font=dict(color='white')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig

def create_enhanced_bar_chart(x, y, title, color='#667eea'):
    """Create an enhanced bar chart"""
    fig = go.Figure(data=[go.Bar(
        x=x, y=y,
        marker_color=color,
        marker_line=dict(color='rgba(255,255,255,0.6)', width=1),
        text=y,
        textposition='auto',
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(color='white', size=16)),
        xaxis=dict(title_font=dict(color='white'), tickfont=dict(color='white')),
        yaxis=dict(title_font=dict(color='white'), tickfont=dict(color='white')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=50, l=50, r=50)
    )
    return fig

# =========================
# Enhanced Main App
# =========================
def main():
    # Add animated title with emoji
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">
            🚀 CommentSense AI Analytics
        </h1>
        <p style="font-size: 1.2rem; opacity: 0.9; animation: fadeIn 2s ease-in;">
            AI-powered analysis of <strong>relevance, sentiment, toxicity, spam</strong> and a <strong>quality-weighted SoE</strong> metric.
        </p>
    </div>
    """, unsafe_allow_html=True)

    analyzer = CommentAnalyzer()

    # ---------- Enhanced Sidebar ----------
    st.sidebar.markdown("### 📊 Data Management")
    uploaded_file = st.sidebar.file_uploader(
        "Upload comments CSV/XLSX", 
        type=["csv", "xlsx"],
        help="Upload your comment data for analysis"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            st.sidebar.success("✅ File uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading file: {e}")
            df = load_sample_data()
    else:
        st.sidebar.info("💡 No file uploaded. Using sample data for demo.")
        df = load_sample_data()

    # Ensure required column
    if "comment" not in df.columns:
        st.error("❌ Dataset must contain a 'comment' column.")
        st.stop()

    # Pick post_text column
    post_text_col = None
    for c in ["caption", "title", "video_caption", "post_text", "text"]:
        if c in df.columns:
            post_text_col = c
            break
    if post_text_col is None:
        if "video_id" in df.columns:
            df[post_text_col := "post_text"] = df["video_id"].astype(str)
        else:
            df[post_text_col := "post_text"] = ""

    # ---------- AI Analysis with enhanced progress ----------
    progress_container = st.empty()
    with progress_container.container():
        st.markdown("### 🤖 AI Analysis in Progress...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔍 Analyzing sentiment...")
        df["sentiment"] = df["comment"].apply(analyzer.analyze_sentiment)
        progress_bar.progress(25)
        
        status_text.text("🛡️ Detecting toxicity...")
        df["toxicity"] = df["comment"].apply(analyzer.detect_toxicity)
        progress_bar.progress(50)
        
        status_text.text("📊 Calculating relevance...")
        df["relevance"] = df.apply(lambda r: analyzer.relevance_to_post(r["comment"], r[post_text_col]), axis=1)
        progress_bar.progress(75)
        
        status_text.text("🏷️ Categorizing content...")
        def cats_or_other(row):
            return analyzer.zero_shot_categories(row["comment"]) if row["relevance"] >= 0.40 else ["other"]
        df["categories"] = df.apply(cats_or_other, axis=1)
        df["quality_score"] = df.apply(lambda r: analyzer.quality_score(r["comment"], r[post_text_col]), axis=1)
        progress_bar.progress(100)
        
        status_text.text("✅ Analysis complete!")
        
    # Clear progress after completion
    progress_container.empty()
    
    # Labels
    df["quality_category"] = pd.cut(df["quality_score"], bins=[-1, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"])
    df["is_spam"] = (df["quality_score"] < 0.25) | (df["toxicity"] > 0.6)

    # ---------- Enhanced Filters ----------
    st.sidebar.markdown("### 🔽 Filters")
    if "video_id" in df.columns:
        vids = st.sidebar.multiselect(
            "📹 Video IDs", 
            df["video_id"].unique().tolist(), 
            default=df["video_id"].unique().tolist()
        )
        df = df[df["video_id"].isin(vids)]
    
    quality_sel = st.sidebar.multiselect(
        "⭐ Quality Level", 
        ["High", "Medium", "Low"], 
        default=["High", "Medium", "Low"]
    )
    df_filtered = df[df["quality_category"].isin(quality_sel)]

    # ---------- Enhanced KPI Row with animations ----------
    st.markdown("### 📈 Key Performance Indicators")
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
        avg_quality = df_filtered['quality_score'].mean() if len(df_filtered) else 0
        st.metric("⚡ Avg Quality Score", f"{avg_quality:.2f}", delta=f"{avg_quality-0.5:.2f}" if avg_quality > 0.5 else None)

    # ---------- Enhanced Charts ----------
    st.markdown("### 📊 Analytics Dashboard")
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Quality", "😊 Sentiment", "🏷️ Categories", "🚫 Spam"])

    with tab1:
        colA, colB = st.columns(2)
        with colA:
            vc = df_filtered["quality_category"].value_counts()
            fig = px.pie(values=vc.values, names=vc.index, title="Quality Distribution")
            st.plotly_chart(style_plotly_chart(fig), use_container_width=True)
        with colB:
            if "video_id" in df_filtered.columns:
                g = df_filtered.groupby(["video_id", "quality_category"]).size().unstack(fill_value=0)
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
            g = df_filtered.groupby(["is_spam", "quality_category"]).size().unstack(fill_value=0)
            fig = px.bar(g, title="Quality Distribution by Spam Status")
            st.plotly_chart(style_plotly_chart(fig), use_container_width=True)

    # ---------- Enhanced Detailed Table ----------
    st.markdown("### 📋 Detailed Comment Analysis")
    with st.expander("🔍 View Detailed Data Table", expanded=False):
        show_cols = ["comment", "quality_score", "quality_category", "sentiment", "toxicity", "relevance", "categories", "is_spam"]
        missing = [c for c in show_cols if c not in df_filtered.columns]
        for m in missing:
            df_filtered[m] = np.nan
        
        # Enhanced dataframe with styling
        styled_df = df_filtered[show_cols].reset_index(drop=True)
        st.dataframe(
            styled_df, 
            use_container_width=True, 
            height=500,
            column_config={
                "quality_score": st.column_config.ProgressColumn(
                    "Quality Score",
                    help="Quality score from 0 to 1",
                    min_value=0,
                    max_value=1,
                    format="%.2f"
                ),
                "toxicity": st.column_config.ProgressColumn(
                    "Toxicity",
                    help="Toxicity score from 0 to 1",
                    min_value=0,
                    max_value=1,
                    format="%.2f"
                ),
                "relevance": st.column_config.ProgressColumn(
                    "Relevance",
                    help="Relevance score from 0 to 1",
                    min_value=0,
                    max_value=1,
                    format="%.2f"
                )
            }
        )

    # ---------- Enhanced Export ----------
    st.markdown("### 📥 Export Results")
    colDL, colInfo = st.columns([1, 2])
    with colDL:
        if st.button("🔄 Prepare CSV Download", type="primary"):
            st.session_state._csv = df.to_csv(index=False)
            st.success("✅ CSV prepared successfully!")
        
        if "_csv" in st.session_state:
            st.download_button(
                "📊 Download Analysis CSV",
                data=st.session_state._csv,
                file_name=f"comment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="secondary"
            )
    
    with colInfo:
        st.info("💡 **Export includes:** All original data plus AI analysis results (sentiment, toxicity, relevance, categories, quality scores)")

    # ---------- Enhanced Insights ----------
    st.markdown("### 💡 AI-Generated Insights & Recommendations")
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
        for i, tip in enumerate(insights):
            if "🎉" in tip or "😊" in tip or "🎯" in tip:
                st.success(tip)
            elif "⚠️" in tip or "🚨" in tip:
                st.error(tip)
            else:
                st.info(tip)
    else:
        st.info("🔍 Upload your data to get personalized insights!")

    # ---------- Enhanced Chat Assistant ----------
    st.markdown("---")
    st.markdown("### 🤖 Chat with CommentSense AI")
    st.markdown("*Ask me about your data, QCR metrics, or get specific examples!*")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 Hi! I'm your CommentSense AI assistant. Ask me things like:\n\n• **\"What is QCR?\"** - Learn about Quality Comment Ratio\n• **\"Show top posts\"** - See your best performing content\n• **\"Find skincare positive examples\"** - Get specific comment examples\n• **\"Why is spam high?\"** - Understand spam patterns\n\nWhat would you like to explore?",
            }
        ]

    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar="🤖" if m["role"] == "assistant" else "👤"):
            st.markdown(m["content"])

    def chat_reply(prompt: str) -> str:
        p = prompt.lower().strip()

        if "qcr" in p or ("quality" in p and ("ratio" in p or "comment" in p)):
            cur_qcr = (df_filtered["quality_category"] == "High").mean() * 100 if len(df_filtered) else 0
            return f"🎯 **Quality Comment Ratio (QCR)** is the percentage of high-quality comments in your dataset.\n\n📊 **Current QCR: {cur_qcr:.1f}%**\n\n🔍 **How we calculate quality:**\n• Relevance to your content (using AI embeddings)\n• Positive sentiment analysis\n• Content length and specificity\n• Absence of spam/toxic language\n\n💡 **Good QCR benchmarks:**\n• 30%+ = Excellent engagement\n• 15-30% = Good engagement\n• <15% = Room for improvement"

        if "top" in p and ("post" in p or "video" in p):
            # Create a copy to avoid modifying the original
            df_temp = df_filtered.copy()
            cols = [c for c in ["likes", "shares", "saves"] if c in df_temp.columns]
            if cols:
                # Handle division by zero
                std_vals = df_temp[cols].std(ddof=0)
                std_vals = std_vals.replace(0, 1)  # Replace 0 std with 1 to avoid division by zero
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
                                format="%.3f"
                            )
                        }
                    )
                    return "🏆 **Top posts by Quality-weighted Share of Engagement (Q-SoE)** shown above!\n\nQ-SoE combines traditional engagement metrics with our AI quality assessment to identify content that generates both high engagement AND meaningful discussion."
                else:
                    return "📊 No video data available for ranking."
            return "🤖 I computed Q-SoE, but couldn't find video groupings in your data."

        if "find" in p or "example" in p:
            want_cat = None
            for c in ["skincare", "fragrance", "makeup", "hair", "other"]:
                if c in p:
                    want_cat = c
                    break
            want_sent = None
            for s in ["positive", "neutral", "negative"]:
                if s in p:
                    want_sent = s.capitalize()
                    break
            
            q = df_filtered.copy()
            if want_cat:
                q = q[q["categories"].apply(lambda L: want_cat in L if isinstance(L, list) else False)]
            if want_sent:
                q = q[q["sentiment"] == want_sent]
            
            q = q.sort_values("quality_score", ascending=False).head(5)
            
            if len(q) > 0:
                display_df = q[["comment", "quality_score", "sentiment", "categories"]].reset_index(drop=True)
                st.dataframe(
                    display_df, 
                    use_container_width=True,
                    column_config={
                        "comment": st.column_config.TextColumn("Comment", width="large"),
                        "quality_score": st.column_config.ProgressColumn(
                            "Quality Score",
                            min_value=0,
                            max_value=1,
                            format="%.2f"
                        )
                    }
                )
                filter_text = f" in **{want_cat}** category" if want_cat else ""
                filter_text += f" with **{want_sent}** sentiment" if want_sent else ""
                return f"🔍 Found **{len(q)}** high-quality examples{filter_text}!\n\nThese comments score highest on our quality metrics, combining relevance, constructiveness, and engagement value."
            else:
                return f"🤷‍♂️ No examples found matching your criteria. Try adjusting the category or sentiment filter."

        if "spam" in p and ("why" in p or "high" in p or "cause" in p):
            spam_rate = (df_filtered["is_spam"]).mean() * 100 if len(df_filtered) else 0
            spam_comments = df_filtered[df_filtered["is_spam"]]
            spam_examples = spam_comments["comment"].head(3).tolist() if len(spam_comments) > 0 else []
            
            spam_details = f"""🚨 **Spam Analysis: {spam_rate:.1f}%** of your comments are flagged as spam.

🔍 **Common spam indicators we detect:**
• Promotional keywords ("subscribe", "check out", "link in bio")
• Very short or repetitive content
• High toxicity scores
• Off-topic comments with low relevance

💡 **Reduce spam by:**
• Setting up keyword filters
• Enabling comment moderation
• Using community guidelines prompts
• Encouraging specific, on-topic questions"""

            if spam_examples:
                spam_details += f"\n\n📋 **Sample spam comments:**\n" + "\n".join([f'• "{comment[:80]}..."' for comment in spam_examples])
            
            return spam_details

        if "help" in p or "what can you do" in p:
            return """🤖 **I'm your CommentSense AI assistant!** Here's what I can help with:

📊 **Analytics Explanations:**
• Explain QCR, Q-SoE, and other metrics
• Break down quality scoring methodology

🏆 **Performance Insights:**
• Identify your top-performing content
• Analyze engagement patterns
• Compare video performance

🔍 **Content Discovery:**
• Find high-quality comment examples
• Filter by category (skincare, fragrance, etc.)
• Sort by sentiment (positive, neutral, negative)

🚨 **Problem Diagnosis:**
• Explain spam patterns
• Identify engagement issues
• Suggest improvement strategies

💬 **Just ask me naturally!** I understand questions like:
• "Why is my engagement low?"
• "Show me positive skincare comments"
• "What makes a quality comment?"
• "How can I reduce spam?"
"""

        # Fallback with suggestions
        return """🤔 I'm not sure about that specific question. Try asking me about:

🎯 **\"What is QCR?\"** - Learn about quality metrics
🏆 **\"Show top posts\"** - See performance rankings  
🔍 **\"Find [category] [sentiment] examples\"** - Get specific examples
🚨 **\"Why is spam high?\"** - Understand content issues
❓ **\"Help\"** - See all my capabilities"""

    if user_prompt := st.chat_input("Ask about QCR, top posts, examples, or anything else..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_prompt)

        with st.spinner("🤖 Analyzing..."):
            reply = chat_reply(user_prompt)
        
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(reply)

    # ---------- Footer ----------
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem; opacity: 0.7;">
            <p>🚀 <strong>CommentSense AI Analytics</strong> | Powered by Transformer Models & Advanced NLP</p>
            <p>Built with ❤️ using Streamlit, HuggingFace Transformers, and Sentence Transformers</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# ---- Enhanced run
if __name__ == "__main__":
    main()