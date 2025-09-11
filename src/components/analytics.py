import streamlit as st
import pandas as pd

def display_analytics(df: pd.DataFrame, analyzer) -> pd.DataFrame:
    """
    Process and display analytics for comment data
    
    Args:
        df (pd.DataFrame): Input DataFrame containing comments
        analyzer (CommentAnalyzer): Instance of comment analyzer
    
    Returns:
        pd.DataFrame: Processed DataFrame with analysis results
    """
    if df.empty or 'comment' not in df.columns:
        st.error("No comments found in the dataset")
        return df

    # Process comments with analyzer
    with st.spinner("Analyzing comments..."):
        df['sentiment'] = df['comment'].apply(analyzer.analyze_sentiment)
        df['quality_score'] = df['comment'].apply(analyzer.calculate_quality_score)
        df['is_spam'] = df['comment'].apply(analyzer.detect_spam)
        df['categories'] = df['comment'].apply(analyzer.categorize_comment)
        
        # Classify quality based on score
        df['quality_category'] = df['quality_score'].apply(
            lambda score: 'High' if score >= 4 else 'Medium' if score >= 2 else 'Low'
        )

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Comments", len(df))
    
    with col2:
        quality_pct = (len(df[df['quality_category'] == 'High']) / len(df) * 100)
        st.metric("Quality Comments %", f"{quality_pct:.1f}%")
    
    with col3:
        spam_pct = (len(df[df['is_spam']]) / len(df) * 100)
        st.metric("Spam Comments %", f"{spam_pct:.1f}%")
    
    with col4:
        avg_score = df['quality_score'].mean()
        st.metric("Avg Quality Score", f"{avg_score:.1f}")

    return df