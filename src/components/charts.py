import sys
import os
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import streamlit as st
import plotly.express as px
import pandas as pd
from collections import Counter

def render_quality_distribution(df):
    quality_counts = df['quality_category'].value_counts()
    fig_quality = px.pie(values=quality_counts.values, 
                          names=quality_counts.index,
                          title="Comment Quality Distribution",
                          color_discrete_map={'High': '#28a745', 'Medium': '#ffc107', 'Low': '#dc3545'})
    st.plotly_chart(fig_quality, use_container_width=True)

def render_sentiment_distribution(df):
    sentiment_counts = df['sentiment'].value_counts()
    fig_sentiment = px.bar(x=sentiment_counts.index, 
                            y=sentiment_counts.values,
                            title="Sentiment Distribution",
                            color=sentiment_counts.index,
                            color_discrete_map={'Positive': '#28a745', 'Neutral': '#6c757d', 'Negative': '#dc3545'})
    st.plotly_chart(fig_sentiment, use_container_width=True)

def render_category_distribution(df):
    all_categories = []
    for cat_list in df['categories']:
        all_categories.extend(cat_list)
    
    if all_categories:
        category_counts = Counter(all_categories)
        fig_categories = px.bar(x=list(category_counts.keys()), 
                                 y=list(category_counts.values()),
                                 title="Comment Categories Distribution")
        st.plotly_chart(fig_categories, use_container_width=True)
    else:
        st.info("No categories found in the filtered data.")

def render_spam_detection_results(df):
    spam_counts = df['is_spam'].value_counts()
    fig_spam = px.pie(values=spam_counts.values,
                      names=['Clean Comments' if not x else 'Spam Comments' for x in spam_counts.index],
                      title="Spam Detection Results",
                      color_discrete_map={'Clean Comments': '#28a745', 'Spam Comments': '#dc3545'})
    st.plotly_chart(fig_spam, use_container_width=True)

def render_quality_vs_spam(df):
    spam_quality = df.groupby(['is_spam', 'quality_category']).size().unstack(fill_value=0)
    fig_spam_quality = px.bar(spam_quality, 
                               title="Quality Distribution: Spam vs Clean Comments",
                               color_discrete_map={'High': '#28a745', 'Medium': '#ffc107', 'Low': '#dc3545'})
    st.plotly_chart(fig_spam_quality, use_container_width=True)

def render_charts(df):
    if df.empty:
        st.info("Upload data to view analytics charts")
        return

    # Analytics Dashboard header and tabs
    st.subheader("Analytics Dashboard")
    tabs = st.tabs(["Quality Analysis", "Sentiment Analysis", "Category Breakdown", "Spam Detection"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Quality distribution pie chart
            quality_counts = df['quality_category'].value_counts()
            fig_quality = px.pie(
                values=quality_counts.values, 
                names=quality_counts.index,
                title="Comment Quality Distribution",
                color_discrete_map={
                    'High': '#28a745',
                    'Medium': '#ffc107',
                    'Low': '#dc3545'
                }
            )
            st.plotly_chart(fig_quality, use_container_width=True)
        
        with col2:
            # Quality by video stacked bar chart
            if 'video_id' in df.columns:
                quality_by_video = df.groupby(['video_id', 'quality_category']).size().unstack(fill_value=0)
                fig_video_quality = px.bar(
                    quality_by_video, 
                    title="Quality Distribution by Video",
                    barmode='stack',
                    color_discrete_map={
                        'High': '#28a745',
                        'Medium': '#ffc107',
                        'Low': '#dc3545'
                    }
                )
                st.plotly_chart(fig_video_quality, use_container_width=True)
    
    with tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment bar chart
            sentiment_counts = df['sentiment'].value_counts()
            fig_sentiment = px.bar(
                x=sentiment_counts.index, 
                y=sentiment_counts.values,
                title="Sentiment Distribution",
                color=sentiment_counts.index,
                color_discrete_map={
                    'Positive': '#28a745',
                    'Neutral': '#6c757d',
                    'Negative': '#dc3545'
                }
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col2:
            # Quality score by sentiment box plot
            if len(df) > 0:
                fig_sentiment_quality = px.box(
                    df, 
                    x='sentiment', 
                    y='quality_score',
                    title="Quality Score by Sentiment"
                )
                st.plotly_chart(fig_sentiment_quality, use_container_width=True)
    
    with tabs[2]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution bar chart
            all_categories = []
            for cat_list in df['categories']:
                all_categories.extend(cat_list)
            
            if all_categories:
                category_counts = Counter(all_categories)
                fig_categories = px.bar(
                    x=list(category_counts.keys()), 
                    y=list(category_counts.values()),
                    title="Comment Categories Distribution"
                )
                st.plotly_chart(fig_categories, use_container_width=True)
        
        with col2:
            # Quality vs Spam stacked bar chart
            spam_quality = df.groupby(['is_spam', 'quality_category']).size().unstack(fill_value=0)
            fig_spam_quality = px.bar(
                spam_quality, 
                title="Quality Distribution: Spam vs Clean Comments",
                barmode='stack',
                color_discrete_map={
                    'High': '#28a745',
                    'Medium': '#ffc107',
                    'Low': '#dc3545'
                }
            )
            st.plotly_chart(fig_spam_quality, use_container_width=True)
    
    with tabs[3]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Spam detection pie chart
            spam_counts = df['is_spam'].value_counts()
            fig_spam = px.pie(
                values=spam_counts.values,
                names=['Clean Comments' if not x else 'Spam Comments' for x in spam_counts.index],
                title="Spam Detection Results",
                color_discrete_map={
                    'Clean Comments': '#28a745',
                    'Spam Comments': '#dc3545'
                }
            )
            st.plotly_chart(fig_spam, use_container_width=True)
        
        with col2:
            # Spam over time line chart
            if 'publishedAt' in df.columns:
                df['date'] = pd.to_datetime(df['publishedAt']).dt.date
                spam_over_time = df.groupby(['date', 'is_spam']).size().unstack(fill_value=0)
                fig_spam_time = px.line(
                    spam_over_time,
                    title='Spam Comments Over Time',
                    color_discrete_map={
                        True: '#dc3545',
                        False: '#28a745'
                    }
                )
                st.plotly_chart(fig_spam_time, use_container_width=True)