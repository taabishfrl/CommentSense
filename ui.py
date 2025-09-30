from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analyzer import CommentAnalyzer
from data import load_sample_data
from utils import detect_comment_column, detect_numeric_column, detect_date_column
from plots import style_plotly_chart

# PM features
from features_adgen import pick_positive_candidates, make_ad_slogans, outreach_template
from features_qual import cluster_themes, summarize_negative_reasons

# Optional word cloud
try:
    from features_wordcloud import make_wordcloud
    HAS_WC = True
except Exception:
    HAS_WC = False


# ---------------------------
# Data loading & auto-mapping
# ---------------------------
def _load_dataframe(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

            # Auto-detect columns
            comment_col = detect_comment_column(df)
            like_col = detect_numeric_column(df, ['likes', 'likeCount', 'favorites', 'engagement', 'score', 'upvotes'])
            date_col = detect_date_column(df)
            id_col = None
            for col in df.columns:
                if any(k in col.lower() for k in ['id', 'video', 'post', 'item', 'product']):
                    id_col = col
                    break
            if id_col is None and len(df.columns) > 1:
                id_col = df.columns[0]

            if comment_col:
                df = df.rename(columns={comment_col: 'comment'})
                st.sidebar.success(f"✅ Using '{comment_col}' as comment column")

            if like_col:
                df = df.rename(columns={like_col: 'likes'})
                st.sidebar.success(f"✅ Using '{like_col}' as likes column")
            else:
                df['likes'] = 0
                st.sidebar.info("ℹ️ No likes column found. Using default value (0).")

            if date_col:
                df = df.rename(columns={date_col: 'timestamp'})
                if df['timestamp'].dtype == 'object':
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    except Exception:
                        df['timestamp'] = pd.Timestamp.now()
                st.sidebar.success(f"✅ Using '{date_col}' as timestamp column")
            else:
                df['timestamp'] = pd.Timestamp.now()
                st.sidebar.info("ℹ️ No timestamp column found. Using current time.")

            if id_col:
                df = df.rename(columns={id_col: 'video_id'})
                st.sidebar.success(f"✅ Using '{id_col}' as video ID column")
            else:
                df['video_id'] = 'Item_' + df.index.astype(str)
                st.sidebar.info("ℹ️ No ID column found. Generating automatic IDs.")

            st.sidebar.subheader("📋 Data Preview")
            st.sidebar.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
            if st.sidebar.checkbox("Show first 3 rows"):
                st.sidebar.dataframe(df.head(3))
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.info("Falling back to sample data for demonstration.")
            return load_sample_data()
    else:
        st.info("📝 No file uploaded. Using sample data for demonstration.")
        return load_sample_data()


def _manual_mapping(df):
    st.sidebar.subheader("🛠️ Manual Column Mapping (Optional)")
    all_columns = df.columns.tolist()
    current_comment_col = 'comment' if 'comment' in df.columns else all_columns[0] if all_columns else None
    if all_columns:
        manual_comment_col = st.sidebar.selectbox(
            "Select comment/text column",
            options=all_columns,
            index=all_columns.index(current_comment_col) if current_comment_col in all_columns else 0
        )
        if manual_comment_col != current_comment_col and manual_comment_col != 'comment':
            df = df.rename(columns={manual_comment_col: 'comment'})
            st.sidebar.info(f"Using '{manual_comment_col}' as comment column")
    if 'comment' not in df.columns:
        st.error("Could not identify a text column in your data.")
        st.info("Please ensure your CSV contains at least one column with text content.")
        return None
    return df


def _ensure_post_text(df):
    # Return existing descriptive column when available, else create post_text
    for c in ["caption", "title", "video_caption", "post_text", "text", "description"]:
        if c in df.columns:
            return c
    if "video_id" in df.columns:
        df["post_text"] = df["video_id"].astype(str)
    else:
        df["post_text"] = ""
    return "post_text"


# -----------------------
# Analysis progress card
# -----------------------
def _analysis_card(df, analyzer: CommentAnalyzer, post_text_col: str):
    card_progress = st.empty()
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
        df["relevance"] = df.apply(
            lambda r: analyzer.relevance_to_post(r["comment"], r.get(post_text_col, "")),
            axis=1
        )
        progress_bar.progress(75)

        status_text.text("Categorizing content...")
        def cats_or_other(row):
            return analyzer.zero_shot_categories(row["comment"]) if row["relevance"] >= 0.40 else ["other"]
        df["categories"] = df.apply(cats_or_other, axis=1)

        df["quality_score"] = df.apply(lambda r: analyzer.quality_score(r["comment"], r.get(post_text_col, "")), axis=1)
        progress_bar.progress(100)

        status_text.text("✅ Analysis complete!")
    card_progress.empty()
    return df


# -----------------
# Header + KPI card
# -----------------
def _video_title_banner(df_filtered):
    if "title" in df_filtered.columns and not df_filtered["title"].dropna().empty:
        unique_titles = df_filtered["title"].dropna().astype(str).unique().tolist()
        if len(unique_titles) == 1:
            subtitle = f"🎬 {unique_titles[0]}"
        else:
            mode_title = df_filtered["title"].mode()[0]
            subtitle = f"🎬 {mode_title} (+{len(unique_titles)-1} more)"
    else:
        subtitle = "🎬 (unknown)"
    st.markdown(f'<div class="cs-video-title">{subtitle}</div>', unsafe_allow_html=True)


def _kpis_card(df_filtered):
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


# -----------------------
# Analytics + Actions + Themes tabs
# -----------------------
def _analytics_card(df_filtered):
    st.markdown('<div class="cs-card-marker"></div>', unsafe_allow_html=True)
    st.markdown("### Analytics Dashboard")

    # Added a dedicated Themes (word cloud) tab at the end
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["🎯 Quality", "😊 Sentiment", "🏷️ Categories", "🚫 Spam", "🧩 Actions", "☁️ Themes"]
    )

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

    # --------------------------
    # Actions tab (PM playbook)
    # --------------------------
    with tab5:
        st.subheader("Ad-ready copy from positive comments")
        pos_cands = pick_positive_candidates(df_filtered)
        if len(pos_cands) >= 1:
            slogans = make_ad_slogans(pos_cands["comment"].tolist(), topk=3)
            st.success("Suggested ad copy:")
            for s in slogans:
                st.write(f"• {s}")
        else:
            st.info("No strong positive comments found yet. Try widening filters.")

        st.markdown("---")
        st.subheader("Outreach draft to creator")
        creator_col = next((c for c in ["author", "username", "user", "creator", "handle", "channel", "user_name"]
                            if c in df_filtered.columns), None)
        sample_handle = None
        if creator_col and len(pos_cands) and pd.notna(pos_cands.iloc[0][creator_col]):
            sample_handle = str(pos_cands.iloc[0][creator_col])
        draft = outreach_template(sample_handle, product_name="our upcoming launch")
        st.text_area("Copy-paste message", value=draft, height=110)

        st.markdown("---")
        st.subheader("Explain viral negative comments and suggest fixes")

        themes = summarize_negative_reasons(df_filtered, like_col="likes", topk_themes=3)
        if themes:
            for t in themes:
                st.warning(f"**{t['theme']}** — {t['count']} comments (avg {t['avg_likes']} likes)")
                st.write(f"Example: “{t['example']}”")
            st.info(
                "Recommended fixes: prioritize the top theme(s) above. "
                "If **Performance/Speed** appears, consider performance optimizations or a Lite mode; "
                "if **Crashes & Bugs**, focus on stability/QA; if **Pricing/Fees**, revisit packaging/discounts."
            )
        else:
            st.info("No clear viral negative themes detected in the current filter.")

    # --------------------------
    # Themes tab (Word Cloud + Clusters)
    # --------------------------
    with tab6:
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Word Cloud")
            if not HAS_WC:
                st.info("Install `wordcloud` to enable this: `pip install wordcloud`")
            else:
                if len(df_filtered) >= 5:
                    # allow a quick sentiment slice for the cloud
                    sent_sel = st.radio(
                        "Text source",
                        ["All", "Positive only", "Negative only"],
                        horizontal=True
                    )
                    if sent_sel == "Positive only":
                        texts = df_filtered[df_filtered["sentiment"] == "Positive"]["comment"].astype(str).tolist()
                    elif sent_sel == "Negative only":
                        texts = df_filtered[df_filtered["sentiment"] == "Negative"]["comment"].astype(str).tolist()
                    else:
                        texts = df_filtered["comment"].astype(str).tolist()

                    if len(texts) >= 3:
                        st.pyplot(make_wordcloud(texts))
                    else:
                        st.info("Not enough text for a word cloud in this slice.")
                else:
                    st.info("Upload more comments to see the word cloud.")
        with right:
            st.subheader("Top Themes (auto-clustered)")
            if len(df_filtered) >= 6:
                res = cluster_themes(df_filtered, k=4)
                labels = res.get("labels", {})
                if labels:
                    for cid, terms in labels.items():
                        st.write(f"• **Theme {cid+1}**: " + ", ".join(terms[:6]))
                else:
                    st.info("No clear clusters found.")
            else:
                st.info("Need ~6+ comments for theme clustering.")

    # ---- Detailed Table + Export (kept) ----
    st.markdown("### Detailed Comment Analysis")
    with st.expander("View Detailed Data Table", expanded=False):
        show_cols = ["comment", "quality_score", "quality_category", "sentiment", "toxicity", "relevance", "categories", "is_spam"]
        missing = [c for c in show_cols if c not in df_filtered.columns]
        for m in missing:
            df_filtered[m] = np.nan
        styled_df = df_filtered[show_cols].reset_index(drop=True)
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=500,
            column_config={
                "quality_score": st.column_config.ProgressColumn("Quality Score", min_value=0, max_value=1, format="%.2f"),
                "toxicity": st.column_config.ProgressColumn("Toxicity", min_value=0, max_value=1, format="%.2f"),
                "relevance": st.column_config.ProgressColumn("Relevance", min_value=0, max_value=1, format="%.2f"),
            },
        )

    st.markdown("### Export Results")
    colDL, colInfo = st.columns([1, 2])
    with colDL:
        st.markdown('<span class="csv-btn-marker"></span>', unsafe_allow_html=True)
        if st.button("Prepare CSV Download", type="primary", key="prep_csv"):
            st.session_state._csv = df_filtered.to_csv(index=False)  # export current filtered view
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
        st.info(" **Export includes:** All data visible in the table plus AI analysis results (sentiment, toxicity, relevance, categories, quality scores)")


# ---------------
# Insights panel
# ---------------
def _insights(df_filtered):
    st.markdown("### AI-Generated Insights & Recommendations")
    insights = []
    if len(df_filtered):
        qcr = (df_filtered["quality_category"] == "High").mean() * 100
        spam = (df_filtered["is_spam"]).mean() * 100
        pos_sentiment = (df_filtered["sentiment"] == "Positive").mean() * 100
        high_toxicity = (df_filtered["toxicity"] > 0.5).mean() * 100
        avg_relevance = df_filtered["relevance"].mean()

        if qcr > 30:
            insights.append("🎉 High QCR detected — your content is sparking meaningful, on-topic discussion.")
        elif qcr < 15:
            insights.append("📈 QCR improvement opportunity — try clearer CTAs or more specific captions to prompt constructive replies.")
        if spam > 20:
            insights.append("🚨 Elevated spam levels — consider moderation rules or blocking promotional keywords.")
        if pos_sentiment > 60:
            insights.append("😊 Positive audience sentiment — consider scaling this content theme.")
        elif pos_sentiment < 30:
            insights.append("😐 Neutral/negative sentiment is high — consider adjusting content approach to increase engagement.")
        if high_toxicity > 10:
            insights.append("⚠️ Toxicity alert — consider stricter comment moderation.")
        if avg_relevance > 0.7:
            insights.append("🎯 High relevance — comments are well aligned with your content topics.")
        elif avg_relevance < 0.4:
            insights.append("🔄 Low relevance — comments may be off-topic. Consider more focused content themes.")
    if insights:
        for tip in insights:
            if any(k in tip for k in ["🎉", "😊", "🎯"]):
                st.success(tip)
            elif any(k in tip for k in ["⚠️", "🚨"]):
                st.error(tip)
            else:
                st.info(tip)
    else:
        st.info("Upload your data to get personalized insights.")


# -----------
# Chat block
# -----------
def _chat_card(df_filtered):
    st.markdown('<div class="cs-card-marker"></div>', unsafe_allow_html=True)
    st.markdown("### 🤖 Chat with CommentSense AI")
    st.markdown("*Ask me about your data, QCR metrics, or get specific examples.*")

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 Hi! I'm your CommentSense AI assistant. Ask me things like:\n\n• **\"What is QCR?\"**\n• **\"Show top posts\"**\n• **\"Find skincare positive examples\"**\n• **\"Why is spam high?\"**\n\nWhat would you like to explore?",
        }]

    from chat_logic import answer
    respond = answer(df_filtered)

    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar="🤖" if m["role"] == "assistant" else "👤"):
            st.markdown(m["content"])

    if user_prompt := st.chat_input("Ask about QCR, top posts, examples, or anything else..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_prompt)
        with st.spinner("🤖 Analyzing..."):
            reply = respond(user_prompt)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(reply)


def _footer():
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


# ----------
# Entrypoint
# ----------
def render_app():
    analyzer = CommentAnalyzer()

    st.sidebar.markdown("#### Data Management")
    uploaded_file = st.sidebar.file_uploader(
        "Upload comments CSV/XLSX", type=["csv", "xlsx"], help="Upload your comment data for analysis"
    )

    df = _load_dataframe(uploaded_file)
    df = _manual_mapping(df)
    if df is None:
        return

    post_text_col = _ensure_post_text(df)
    df = _analysis_card(df, analyzer, post_text_col)

    df["quality_category"] = pd.cut(df["quality_score"], bins=[-1, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"])
    df["is_spam"] = (df["quality_score"] < 0.25) | (df["toxicity"] > 0.6)

    st.sidebar.markdown("#### Filters")
    df_work = df.copy()
    if "video_id" in df_work.columns:
        vids = st.sidebar.multiselect(
            "📹 Video IDs",
            df_work["video_id"].unique().tolist(),
            default=df_work["video_id"].unique().tolist()
        )
        df_work = df_work[df_work["video_id"].isin(vids)]

    quality_sel = st.sidebar.multiselect("⭐ Quality Level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    df_filtered = df_work[df_work["quality_category"].isin(quality_sel)]

    _video_title_banner(df_filtered)

    # Card 1: KPIs
    with st.container():
        _kpis_card(df_filtered)

    # Card 2: Analytics + Actions + Themes + Table + Export + Insights
    with st.container():
        st.markdown('<div class="cs-card-marker"></div>', unsafe_allow_html=True)
        _analytics_card(df_filtered)
        _insights(df_filtered)

    # Card 3: Chat
    with st.container():
        _chat_card(df_filtered)

    _footer()
