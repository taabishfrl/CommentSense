import numpy as np
import pandas as pd
import streamlit as st

def answer(df_filtered: pd.DataFrame):
    def _top_posts_table(df_temp: pd.DataFrame):
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

    def _find_examples(dfq: pd.DataFrame, want_cat, want_sent):
        q = dfq.copy()
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
            if want_sent:
                filter_text += f" with **{want_sent}** sentiment"
            return (f"🔍 Found **{len(q)}** high-quality examples{filter_text}!\n\n"
                    "These comments score highest on our quality metrics, combining relevance, constructiveness, and engagement value.")
        else:
            return "🤷‍♂️ No examples found matching your criteria. Try adjusting the category or sentiment filter."

    def respond(prompt: str) -> str:
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
            return _top_posts_table(df_temp)

        if "find" in p or "example" in p:
            want_cat = None
            for c in ["skincare","fragrance","makeup","hair","other"]:
                if c in p:
                    want_cat = c; break
            want_sent = None
            for s in ["positive","neutral","negative"]:
                if s in p:
                    want_sent = s.capitalize(); break
            return _find_examples(df_filtered, want_cat, want_sent)

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
                spam_details += "\n\n **Sample spam comments:**\n" + "\n".join([f'• "{c[:80]}..."' for c in spam_examples])
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

    return respond
