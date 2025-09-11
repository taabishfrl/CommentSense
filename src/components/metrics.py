import streamlit as st
import pandas as pd

def display_metrics(df):
    if not df.empty:
        # Engagement metrics using actual column names from your CSV
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'likeCount' in df.columns:  # Changed from 'likes' to 'likeCount'
                avg_likes = df['likeCount'].mean()
                st.metric("Average Likes", f"{avg_likes:.1f}")
            else:
                st.metric("Average Likes", "N/A")
            
        with col2:
            if 'parentCommentId' in df.columns:  # This can be used to count replies
                reply_count = df['parentCommentId'].notna().sum()
                st.metric("Total Replies", f"{reply_count}")
            else:
                st.metric("Total Replies", "N/A")
            
        with col3:
            if 'likeCount' in df.columns:
                total_comments = len(df)
                total_likes = df['likeCount'].sum()
                engagement_rate = (total_likes / total_comments) * 100 if total_comments > 0 else 0
                st.metric("Engagement Rate", f"{engagement_rate:.1f}%")
            else:
                st.metric("Engagement Rate", "N/A")