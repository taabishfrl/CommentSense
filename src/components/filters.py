# filepath: /my-streamlit-app/my-streamlit-app/src/components/filters.py

import streamlit as st
import pandas as pd

def apply_filters(load_sample_data):
    """
    Apply filters to the dataset
    Args:
        load_sample_data: Function that returns sample DataFrame if no file is uploaded
    Returns:
        DataFrame: Filtered dataframe
    """
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Comment Dataset", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            df = load_sample_data()
    else:
        st.info("No file uploaded. Using sample data for demonstration.")
        df = load_sample_data()
    
    if df.empty:
        return df
        
    # Video ID filter
    if 'video_id' in df.columns:
        video_ids = df['video_id'].unique()
        selected_videos = st.sidebar.multiselect(
            "Select Video IDs",
            options=video_ids,
            default=video_ids
        )
        if selected_videos:
            df = df[df['video_id'].isin(selected_videos)]
    
    # Quality Level filter (if exists)
    if 'quality_category' in df.columns:
        quality_levels = ['High', 'Medium', 'Low']
        selected_quality = st.sidebar.multiselect(
            "Quality Level",
            options=quality_levels,
            default=quality_levels
        )
        if selected_quality:
            df = df[df['quality_category'].isin(selected_quality)]
    
    return df