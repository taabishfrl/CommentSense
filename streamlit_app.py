# CommentSense — Streamlit end-to-end demo (free, no paid APIs)
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
# Page + header styling
# =========================
st.set_page_config(
    page_title="CommentSense AI Analytics",
    page_icon="logo.png",  # favicon only
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
/* make header a reliable positioning parent */
[data-testid="stHeader"] {{
  position: relative;
  background:#f8f9fa !important;
  height:72px;
}}
            
/* center a big logo as a pseudo element */
[data-testid="stHeader"]::before {{
  content: "";
  position: absolute;
  top: 50%;
  left: 50%;
  width: 200px;                /* <-- adjust width */
  height: 56px;                /* <-- adjust height */
  transform: translate(-50%, -50%);
  background-image: url("{LOGO_URI}");
  background-repeat: no-repeat;
  background-position: center center;
  background-size: contain;
  z-index: 5;                  /* below buttons but above bg */
  pointer-events: none;        /* clicks pass through */
}}

/* keep toolbar transparent so the logo shows */
[data-testid="stToolbar"], [data-testid="stToolbarActions"] {{
  background: transparent !important;
}}

/* Target the chat input container */
[data-testid="stChatInput"] {{
    width: 100%;
    background: transparent;
    padding-bottom: 3rem;
}}

/* Apply padding to the main app container */
.stMainBlockContainer {{
    padding-top: 1rem !important;  
}}

</style>
""", unsafe_allow_html=True)

# =========================
# Cached model loaders
# =========================
@st.cache_resource
def load_embedder():
    # Multilingual option: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_sentiment():
    # Multilingual option: "nlptown/bert-base-multilingual-uncased-sentiment"
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

@st.cache_resource
def load_zero_shot():
    # Lighter multilingual: "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@st.cache_resource
def load_toxicity():
    # Free toxicity classifier (no Detoxify/Torch-hub checkpoints)
    # We request all scores so we can extract the "toxic" probability.
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
# Analyzer (AI + light rules)
# =========================
class CommentAnalyzer:
    def __init__(self):
        self.embedder = load_embedder()
        self.sentiment_pipe = load_sentiment()
        self.zs_pipe = load_zero_shot()
        self.tox_pipe = load_toxicity()

        # Lightweight bonuses/penalties
        self.quality_keywords = {
            "high_quality": [
                "insightful",
                "helpful",
                "informative",
                "detailed",
                "thoughtful",
                "constructive",
                "in-depth",
                "compare",
                "recommend",
                "results",
                "review",
            ],
            "spam_indicators": [
                "subscribe",
                "check out",
                "visit my",
                "link in bio",
                "click here",
                "follow me",
                "dm me",
                "whatsapp",
                "promo",
                "discount",
                "coupon",
            ],
        }
        self.categories = ["skincare", "fragrance", "makeup", "hair", "other"]

    # --- AI pieces ---
    def analyze_sentiment(self, comment: str) -> str:
        try:
            out = self.sentiment_pipe(str(comment))[0]["label"].upper()
            return {"NEGATIVE": "Negative", "NEUTRAL": "Neutral", "POSITIVE": "Positive"}.get(out, "Neutral")
        except Exception:
            return "Neutral"

    def detect_toxicity(self, comment: str) -> float:
        """Return probability the comment is toxic (0..1) via HF pipeline."""
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
        sim = float((emb[0] * emb[1]).sum())  # cosine (normalized)
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))  # map [-1,1] -> [0,1]

    # --- Hybrid QualityScore (0..1) ---
    def quality_score(self, comment: str, post_text: str) -> float:
        if not isinstance(comment, str) or not comment.strip():
            return 0.0
        c = comment.lower()

        rel = self.relevance_to_post(comment, post_text)  # 0..1
        tox = self.detect_toxicity(comment)  # 0..1
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
# Sample data for demo
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
# App
# =========================
def main():
    st.markdown("AI-powered analysis of **relevance, sentiment, toxicity, spam** and a **quality-weighted SoE** metric.")

    analyzer = CommentAnalyzer()

    # ---------- Sidebar: data upload ----------
    st.sidebar.title("Data")
    uploaded_file = st.sidebar.file_uploader("Upload any CSV/XLSX file", type=["csv", "xlsx"])
    
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

    # pick a post_text column for relevance
    post_text_col = None
    for c in ["caption", "title", "video_caption", "post_text", "text", "description"]:
        if c in df.columns:
            post_text_col = c
            break
    if post_text_col is None:
        if "video_id" in df.columns:
            df[post_text_col := "post_text"] = df["video_id"].astype(str)
        else:
            df[post_text_col := "post_text"] = ""  # fallback

    # ---------- AI analysis ----------
    with st.spinner("Analyzing with free AI models…"):
        df["sentiment"] = df["comment"].apply(analyzer.analyze_sentiment)
        df["toxicity"] = df["comment"].apply(analyzer.detect_toxicity)
        df["relevance"] = df.apply(lambda r: analyzer.relevance_to_post(r["comment"], r[post_text_col]), axis=1)

        # Only run zero-shot on likely on-topic to save RAM/time
        def cats_or_other(row):
            return analyzer.zero_shot_categories(row["comment"]) if row["relevance"] >= 0.40 else ["other"]

        df["categories"] = df.apply(cats_or_other, axis=1)
        df["quality_score"] = df.apply(lambda r: analyzer.quality_score(r["comment"], r[post_text_col]), axis=1)

        # Labels
        df["quality_category"] = pd.cut(df["quality_score"], bins=[-1, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"])
        df["is_spam"] = (df["quality_score"] < 0.25) | (df["toxicity"] > 0.6)

    # ---------- Filters ----------
    st.sidebar.subheader("Filters")
    if "video_id" in df.columns:
        vids = st.sidebar.multiselect("Video IDs", df["video_id"].unique().tolist(), default=df["video_id"].unique().tolist())
        df = df[df["video_id"].isin(vids)]
    quality_sel = st.sidebar.multiselect("Quality", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    df_filtered = df[df["quality_category"].isin(quality_sel)]

    # ---------- KPI row ----------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Comments", len(df_filtered))
    with c2:
        qcr = (df_filtered["quality_category"] == "High").mean() * 100 if len(df_filtered) else 0
        st.metric("QCR (Quality %)", f"{qcr:.1f}%")
    with c3:
        spam = (df_filtered["is_spam"]).mean() * 100 if len(df_filtered) else 0
        st.metric("Spam %", f"{spam:.1f}%")
    with c4:
        st.metric("Avg QualityScore", f"{df_filtered['quality_score'].mean():.2f}" if len(df_filtered) else "0.00")

    # ---------- Charts ----------
    st.subheader("Analytics")
    tab1, tab2, tab3, tab4 = st.tabs(["Quality", "Sentiment", "Categories", "Spam"])

    with tab1:
        colA, colB = st.columns(2)
        with colA:
            vc = df_filtered["quality_category"].value_counts()
            fig = px.pie(values=vc.values, names=vc.index, title="Quality distribution")
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            if "video_id" in df_filtered.columns:
                g = df_filtered.groupby(["video_id", "quality_category"]).size().unstack(fill_value=0)
                fig2 = px.bar(g, title="Quality by video")
                st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        colA, colB = st.columns(2)
        with colA:
            sc = df_filtered["sentiment"].value_counts()
            fig = px.bar(x=sc.index, y=sc.values, title="Sentiment distribution", labels={"x": "Sentiment", "y": "Count"})
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            if len(df_filtered):
                fig = px.box(df_filtered, x="sentiment", y="quality_score", title="QualityScore by sentiment")
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        all_cats = []
        for L in df_filtered["categories"]:
            all_cats.extend(L)
        if all_cats:
            counts = Counter(all_cats)
            fig = px.bar(x=list(counts.keys()), y=list(counts.values()), title="Category counts")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categories detected in current filter.")

    with tab4:
        colA, colB = st.columns(2)
        with colA:
            spam_counts = df_filtered["is_spam"].value_counts()
            names = ["Clean" if not k else "Spam" for k in spam_counts.index]
            fig = px.pie(values=spam_counts.values, names=names, title="Spam vs Clean")
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            g = df_filtered.groupby(["is_spam", "quality_category"]).size().unstack(fill_value=0)
            fig = px.bar(g, title="Quality by Spam/Clean")
            st.plotly_chart(fig, use_container_width=True)

    # ---------- Detailed table ----------
    st.subheader("Detailed Comment Analysis")
    if st.checkbox("Show table"):
        show_cols = ["comment", "quality_score", "quality_category", "sentiment", "toxicity", "relevance", "categories", "is_spam"]
        missing = [c for c in show_cols if c not in df_filtered.columns]
        for m in missing:
            df_filtered[m] = np.nan
        st.dataframe(df_filtered[show_cols].reset_index(drop=True), use_container_width=True, height=500)

    # ---------- Export ----------
    colDL, _ = st.columns([1, 3])
    with colDL:
        if st.button("Prepare CSV"):
            st.session_state._csv = df.to_csv(index=False)
        if "_csv" in st.session_state:
            st.download_button(
                "Download analysis CSV",
                data=st.session_state._csv,
                file_name=f"comment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    # ---------- Insights ----------
    st.subheader("AI Insights & Recommendations")
    insights = []
    if len(df_filtered):
        if qcr > 30:
            insights.append("High QCR — your content is sparking meaningful, on-topic discussion.")
        elif qcr < 15:
            insights.append("Low QCR — try clearer CTAs or more specific captions to prompt constructive replies.")
        if spam > 20:
            insights.append("Spam is elevated — many short/promo/duplicate comments. Consider moderation rules or blocking promo keywords.")
        pos = (df_filtered["sentiment"] == "Positive").mean() * 100
        if pos > 60:
            insights.append("Audience sentiment is strongly positive — consider scaling this content theme.")
    for tip in insights:
        st.info(tip)

    # ---------- Chat assistant (dataset-aware) ----------
    st.divider()
    st.subheader("Chat with CommentSense")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! Ask me things like: **what is QCR**, **show top posts**, **find skincare positive examples**, or **why is spam high**.",
            }
        ]

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    def chat_reply(prompt: str) -> str:
        p = prompt.lower().strip()

        if "qcr" in p or ("quality" in p and "ratio" in p):
            cur_qcr = (df["quality_category"] == "High").mean() * 100 if len(df) else 0
            return f"**QCR** is the share of quality comments (High). Current QCR = **{cur_qcr:.1f}%**. We rate quality via relevance (MiniLM), sentiment, length/specificity bonuses, and toxicity/spam penalties."

        if "top" in p and ("post" in p or "video" in p):
            cols = [c for c in ["likes", "shares", "saves"] if c in df.columns]
            if cols:
                z = (df[cols] - df[cols].mean()) / df[cols].std(ddof=0)
                df["_soe"] = z.mean(axis=1)
            else:
                df["_soe"] = 0.0
            df["_qsoe"] = df["_soe"] * df["quality_score"]
            if "video_id" in df.columns:
                top = df.groupby("video_id")["_qsoe"].mean().sort_values(ascending=False).head(5)
                st.dataframe(top.rename("Q-SoE").reset_index(), use_container_width=True)
                return "Top posts by **Q-SoE** shown above."
            return "I computed Q-SoE, but couldn't find a 'video_id' to group by."

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
            q = df.copy()
            if want_cat:
                q = q[q["categories"].apply(lambda L: want_cat in L)]
            if want_sent:
                q = q[q["sentiment"] == want_sent]
            q = q.sort_values("quality_score", ascending=False).head(5)
            if len(q):
                st.dataframe(q[["comment", "quality_score", "sentiment", "categories"]].reset_index(drop=True), use_container_width=True)
                return f"Showing **{len(q)}** examples{f' in {want_cat}' if want_cat else ''}{f' with {want_sent} sentiment' if want_sent else ''}."
            return "No examples matched your request."

        if "spam" in p and ("why" in p or "high" in p):
            spam_rate = (df["is_spam"]).mean() * 100 if len(df) else 0
            tips = "- Many short/duplicate or promo-keyword comments\n- Toxic language\n- Off-topic replies"
            return f"Spam is **{spam_rate:.1f}%**. We flag spam via low QualityScore, promo keywords, and toxicity.\n\n**Reduce it by:**\n{tips}"

        if "help" in p or "what can you do" in p:
            return "I can explain **QCR/Q-SoE**, rank **top posts**, and surface **example comments** filtered by category or sentiment."

        return "Try: **show top posts**, **find skincare positive examples**, or **what is QCR**."

    if user_prompt := st.chat_input("Ask about QCR, top posts, examples…"):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        reply = chat_reply(user_prompt)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

# ---- run
if __name__ == "__main__":
    main()