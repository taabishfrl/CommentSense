import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import re
from collections import Counter
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

#TESTING 12 12
# Set page config
st.set_page_config(
    page_title="CommentSense AI Analytics",
    page_icon="CSAA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
}
.quality-high { color: #28a745; font-weight: bold; }
.quality-medium { color: #ffc107; font-weight: bold; }
.quality-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class CommentAnalyzer:
    def __init__(self):
        # Define quality indicators
        self.quality_keywords = {
            'high_quality': [
                'insightful', 'helpful', 'informative', 'detailed', 'thoughtful', 
                'well-explained', 'comprehensive', 'valuable', 'educational', 'clear',
                'professional', 'constructive', 'analytical', 'in-depth'
            ],
            'engagement_words': [
                'question', 'why', 'how', 'what', 'when', 'where', 'discuss',
                'thoughts', 'opinion', 'agree', 'disagree', 'interesting', 'curious'
            ],
            'spam_indicators': [
                'subscribe', 'check out', 'visit my', 'link in bio', 'click here',
                'follow me', 'dm me', 'first', 'early', 'notification squad'
            ]
        }
        
        # Category keywords for filtering
        self.category_keywords = {
            'skincare': ['skincare', 'acne', 'moisturizer', 'serum', 'cleanser', 'SPF', 'routine'],
            'fragrance': ['perfume', 'cologne', 'scent', 'fragrance', 'smell', 'notes'],
            'makeup': ['makeup', 'foundation', 'lipstick', 'eyeshadow', 'mascara', 'concealer'],
            'tech': ['technology', 'software', 'hardware', 'app', 'device', 'digital'],
            'gaming': ['game', 'gaming', 'player', 'console', 'PC', 'mobile gaming']
        }

    def analyze_sentiment(self, comment):
        """Analyze sentiment of a comment"""
        try:
            blob = TextBlob(str(comment))
            sentiment = blob.sentiment.polarity
            if sentiment > 0.1:
                return 'Positive'
            elif sentiment < -0.1:
                return 'Negative'
            else:
                return 'Neutral'
        except:
            return 'Neutral'

    def calculate_quality_score(self, comment):
        """Calculate quality score based on multiple factors"""
        if pd.isna(comment):
            return 0
        
        comment_lower = str(comment).lower()
        score = 0
        
        # Length factor (optimal range: 20-200 characters)
        length = len(comment_lower)
        if 20 <= length <= 200:
            score += 2
        elif length > 200:
            score += 1
        
        # High quality keywords
        for word in self.quality_keywords['high_quality']:
            if word in comment_lower:
                score += 2
        
        # Engagement indicators
        for word in self.quality_keywords['engagement_words']:
            if word in comment_lower:
                score += 1
        
        # Proper grammar indicators
        if '?' in comment or '!' in comment:
            score += 1
        
        # Penalty for spam indicators
        for spam_word in self.quality_keywords['spam_indicators']:
            if spam_word in comment_lower:
                score -= 3
        
        # Penalty for excessive caps or repetitive characters
        if len(re.findall(r'[A-Z]', comment)) > len(comment) * 0.3:
            score -= 2
        
        if re.search(r'(.)\1{2,}', comment):  # Repetitive characters
            score -= 1
        
        return max(0, score)  # Ensure non-negative score

    def categorize_comment(self, comment):
        """Categorize comment based on keywords"""
        if pd.isna(comment):
            return 'Uncategorized'
        
        comment_lower = str(comment).lower()
        categories = []
        
        for category, keywords in self.category_keywords.items():
            if any(keyword in comment_lower for keyword in keywords):
                categories.append(category)
        
        return categories if categories else ['Uncategorized']

    def detect_spam(self, comment):
        """Detect if comment is likely spam"""
        if pd.isna(comment):
            return False
        
        comment_lower = str(comment).lower()
        spam_count = sum(1 for spam_word in self.quality_keywords['spam_indicators'] 
                        if spam_word in comment_lower)
        
        # Additional spam indicators
        if len(comment_lower) < 5:  # Too short
            return True
        if spam_count >= 2:  # Multiple spam indicators
            return True
        if re.search(r'(.)\1{4,}', comment):  # Excessive repetition
            return True
        
        return False

def load_sample_data():
    """Generate sample data for demonstration"""
    sample_comments = [
        "This tutorial was incredibly helpful! The step-by-step breakdown made it so easy to follow.",
        "Great video! Could you do one about advanced techniques?",
        "First! Love your content!",
        "This is amazing, really detailed explanation. Thank you!",
        "Subscribe to my channel for more content like this!",
        "I disagree with your approach, but I appreciate the thorough analysis.",
        "Wow!!! So good!!!",
        "Very informative video. The examples you provided were particularly useful for understanding the concept.",
        "Can you please make a video about skincare routine for sensitive skin?",
        "Check out my latest video! Link in bio!",
        "This fragrance sounds interesting. What are the main notes?",
        "Your makeup tutorial always inspire me to try new looks.",
        "The gaming review was spot on. Have you tried the new update?",
        "Thanks for the tech review. Very comprehensive analysis.",
        "aaaaaaaaawesome video!!!!"
    ]
    
    video_ids = ['VID001', 'VID002', 'VID003', 'VID001', 'VID002', 'VID003', 'VID001', 'VID002', 'VID003', 'VID001', 'VID002', 'VID003', 'VID001', 'VID002', 'VID003']
    
    return pd.DataFrame({
        'video_id': video_ids,
        'comment': sample_comments,
        'likes': np.random.randint(0, 50, len(sample_comments)),
        'timestamp': pd.date_range('2024-01-01', periods=len(sample_comments), freq='H')
    })

def main():
    st.markdown('<h1 class="main-header">CommentSense AI Analytics</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Comment Quality Analysis & Share of Engagement Metrics")
    
    # Initialize analyzer
    analyzer = CommentAnalyzer()
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Comment Dataset", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            df = load_sample_data()
    else:
        st.info("No file uploaded. Using sample data for demonstration.")
        df = load_sample_data()
    
    # Data preprocessing
    if 'comment' not in df.columns:
        st.error("Dataset must contain a 'comment' column")
        return
    
    # Analyze comments
    with st.spinner("AI is analyzing comments..."):
        df['sentiment'] = df['comment'].apply(analyzer.analyze_sentiment)
        df['quality_score'] = df['comment'].apply(analyzer.calculate_quality_score)
        df['is_spam'] = df['comment'].apply(analyzer.detect_spam)
        df['categories'] = df['comment'].apply(analyzer.categorize_comment)
        
        # Quality classification
        def classify_quality(score):
            if score >= 4:
                return 'High'
            elif score >= 2:
                return 'Medium'
            else:
                return 'Low'
        
        df['quality_category'] = df['quality_score'].apply(classify_quality)
    
    # Sidebar filters
    st.sidebar.subheader("Filters")
    
    if 'video_id' in df.columns:
        selected_videos = st.sidebar.multiselect("Select Video IDs", 
                                                options=df['video_id'].unique(),
                                                default=df['video_id'].unique())
        df = df[df['video_id'].isin(selected_videos)]
    
    quality_filter = st.sidebar.multiselect("Quality Level", 
                                           options=['High', 'Medium', 'Low'],
                                           default=['High', 'Medium', 'Low'])
    df_filtered = df[df['quality_category'].isin(quality_filter)]
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_comments = len(df_filtered)
        st.metric("Total Comments", total_comments)
    
    with col2:
        quality_ratio = len(df_filtered[df_filtered['quality_category'] == 'High']) / len(df_filtered) * 100
        st.metric("Quality Comments %", f"{quality_ratio:.1f}%")
    
    with col3:
        spam_ratio = len(df_filtered[df_filtered['is_spam'] == True]) / len(df_filtered) * 100
        st.metric("Spam Comments %", f"{spam_ratio:.1f}%")
    
    with col4:
        avg_quality_score = df_filtered['quality_score'].mean()
        st.metric("Avg Quality Score", f"{avg_quality_score:.1f}")
    
    # Charts section
    st.subheader("Analytics Dashboard")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Quality Analysis", "Sentiment Analysis", "Category Breakdown", "Spam Detection"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Quality distribution
            quality_counts = df_filtered['quality_category'].value_counts()
            fig_quality = px.pie(values=quality_counts.values, 
                               names=quality_counts.index,
                               title="Comment Quality Distribution",
                               color_discrete_map={'High': '#28a745', 'Medium': '#ffc107', 'Low': '#dc3545'})
            st.plotly_chart(fig_quality, use_container_width=True)
        
        with col2:
            # Quality by video
            if 'video_id' in df_filtered.columns:
                quality_by_video = df_filtered.groupby(['video_id', 'quality_category']).size().unstack(fill_value=0)
                fig_video_quality = px.bar(quality_by_video, 
                                         title="Quality Distribution by Video",
                                         color_discrete_map={'High': '#28a745', 'Medium': '#ffc107', 'Low': '#dc3545'})
                st.plotly_chart(fig_video_quality, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution
            sentiment_counts = df_filtered['sentiment'].value_counts()
            fig_sentiment = px.bar(x=sentiment_counts.index, 
                                 y=sentiment_counts.values,
                                 title="Sentiment Distribution",
                                 color=sentiment_counts.index,
                                 color_discrete_map={'Positive': '#28a745', 'Neutral': '#6c757d', 'Negative': '#dc3545'})
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col2:
            # Sentiment vs Quality
            if len(df_filtered) > 0:
                fig_sentiment_quality = px.box(df_filtered, 
                                             x='sentiment', 
                                             y='quality_score',
                                             title="Quality Score by Sentiment")
                st.plotly_chart(fig_sentiment_quality, use_container_width=True)
    
    with tab3:
        # Category analysis
        all_categories = []
        for cat_list in df_filtered['categories']:
            all_categories.extend(cat_list)
        
        if all_categories:
            category_counts = Counter(all_categories)
            fig_categories = px.bar(x=list(category_counts.keys()), 
                                  y=list(category_counts.values()),
                                  title="Comment Categories Distribution")
            st.plotly_chart(fig_categories, use_container_width=True)
        else:
            st.info("No categories found in the filtered data.")
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            # Spam detection results
            spam_counts = df_filtered['is_spam'].value_counts()
            fig_spam = px.pie(values=spam_counts.values,
                            names=['Clean Comments' if not x else 'Spam Comments' for x in spam_counts.index],
                            title="Spam Detection Results",
                            color_discrete_map={'Clean Comments': '#28a745', 'Spam Comments': '#dc3545'})
            st.plotly_chart(fig_spam, use_container_width=True)
        
        with col2:
            # Quality vs Spam
            spam_quality = df_filtered.groupby(['is_spam', 'quality_category']).size().unstack(fill_value=0)
            fig_spam_quality = px.bar(spam_quality, 
                                    title="Quality Distribution: Spam vs Clean Comments",
                                    color_discrete_map={'High': '#28a745', 'Medium': '#ffc107', 'Low': '#dc3545'})
            st.plotly_chart(fig_spam_quality, use_container_width=True)
    
    # Detailed view
    st.subheader("Detailed Comment Analysis")
    
    # Show sample comments with analysis
    if st.checkbox("Show Detailed Comment Analysis"):
        display_df = df_filtered[['comment', 'quality_score', 'quality_category', 'sentiment', 'is_spam']].copy()
        
        # Apply styling based on quality
        def style_quality(val):
            if val == 'High':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'Medium':
                return 'background-color: #fff3cd; color: #856404'
            else:
                return 'background-color: #f8d7da; color: #721c24'
        
        styled_df = display_df.style.applymap(style_quality, subset=['quality_category'])
        st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Export functionality
    if st.button("Export Analysis Results"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"comment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Insights and recommendations
    st.subheader("AI Insights & Recommendations")
    
    insights = []
    
    # Quality insights
    high_quality_pct = len(df_filtered[df_filtered['quality_category'] == 'High']) / len(df_filtered) * 100
    if high_quality_pct > 30:
        insights.append("Great engagement! High percentage of quality comments indicates strong audience connection.")
    elif high_quality_pct < 15:
        insights.append("Consider improving content to encourage more meaningful discussions.")
    
    # Spam insights
    spam_pct = len(df_filtered[df_filtered['is_spam'] == True]) / len(df_filtered) * 100
    if spam_pct > 20:
        insights.append("High spam rate detected. Consider implementing stricter moderation.")
    
    # Sentiment insights
    positive_pct = len(df_filtered[df_filtered['sentiment'] == 'Positive']) / len(df_filtered) * 100
    if positive_pct > 60:
        insights.append("Positive audience sentiment! Your content resonates well with viewers.")
    
    for insight in insights:
        st.info(insight)

if __name__ == "__main__":
    main()