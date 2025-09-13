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
    page_icon="assets/mini.png",
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

/* Header styling */
[data-testid="stHeader"] {{
    position: relative;
    background: white !important;
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
    background-position: center;
    background-size: contain;
    z-index: 5;
    pointer-events: none;
}}
[data-testid="stSidebarContent"] {{ background: white; }}

/* ---- MAIN WRAPPER: make transparent, let CARDS provide glass look ---- */
.block-container {{
  width: 70%;
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
@keyframes titlePulse {{
  from {{ text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
  to   {{ text-shadow: 2px 2px 20px rgba(255,255,255,0.6); }}
}}
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
.stTabs [data-baseweb="tab-list"] {{
  gap: 2px; background: rgba(255,255,255,0.1) !important; backdrop-filter: blur(10px) !important;
  border-radius: 25px !important; padding: 4px !important;
}}
.stTabs [data-baseweb="tab"] {{
  background: transparent !important; color: white !important; border-radius: 20px !important;
  padding: 8px 16px !important; transition: all .3s ease !important;
}}
.stTabs [aria-selected="true"] {{ background: linear-gradient(45deg,#667eea,#764ba2) !important; box-shadow: 0 4px 15px rgba(102,126,234,.4) !important; }}

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
            st.sidebar.success("File uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
            df = load_sample_data()
    else:
        st.sidebar.info("No file uploaded. Using sample data for demo.")
        df = load_sample_data()

    # ---------- Guard: required column ----------
    if "comment" not in df.columns:
        st.error("Dataset must contain a 'comment' column.")
        st.stop()

    # ---------- Pick/prepare post_text column ----------
    post_text_col = None
    for c in ["caption","title","video_caption","post_text","text"]:
        if c in df.columns:
            post_text_col = c
            break
    if post_text_col is None:
        if "video_id" in df.columns:
            df[post_text_col := "post_text"] = df["video_id"].astype(str)
        else:
            df[post_text_col := "post_text"] = ""

    # ---------- AI Analysis with progress ----------
    progress_container = st.empty()
    with progress_container.container():
        st.markdown("### AI Analysis in Progress...")
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
    progress_container.empty()

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
            if st.button("Prepare CSV Download", type="primary"):
                st.session_state._csv = df.to_csv(index=False)
                st.success("CSV prepared successfully!")
            if "_csv" in st.session_state:
                st.download_button(
                    "Download Analysis CSV",
                    data=st.session_state._csv,
                    file_name=f"comment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    type="secondary",
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
