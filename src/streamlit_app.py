import streamlit as st
from components.analytics import display_analytics
from components.charts import render_charts
from components.filters import apply_filters
from components.metrics import display_metrics
from config.theme import set_theme
from services.analyzer import CommentAnalyzer
from utils.data_loader import load_sample_data

def main():
    set_theme()
    
    st.title("CommentSense AI Analytics")
    st.write("AI-Powered Comment Quality Analysis & Share of Engagement Metrics")
    
    analyzer = CommentAnalyzer()
    
    # Sidebar for data upload and filters
    st.sidebar.header("Dashboard Controls")
    filtered_df = apply_filters(load_sample_data)
    
    if not filtered_df.empty:
        # Process and display data
        processed_df = display_analytics(filtered_df, analyzer)
        display_metrics(processed_df)
        render_charts(processed_df)
    else:
        st.info("Upload data to begin analysis")

if __name__ == "__main__":
    main()