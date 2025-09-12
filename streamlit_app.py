import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import re
from collections import Counter
import numpy as np
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="CommentSense AI Analytics",
    page_icon="💬",
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
                'professional', 'constructive', 'analytical', 'in-depth', 'useful',
                'great', 'awesome', 'amazing', 'perfect', 'excellent', 'love this',
                'thank you', 'thanks', 'appreciate', 'helpful', 'worked', 'good',
                'recommend', 'solved', 'understand', 'explanation', 'example'
            ],
            'engagement_words': [
                'question', 'why', 'how', 'what', 'when', 'where', 'discuss',
                'thoughts', 'opinion', 'agree', 'disagree', 'interesting', 'curious',
                'can you', 'should i', 'which one', 'compare', 'difference', 'best',
                'recommendation', 'suggestion', 'advice', 'tip'
            ],
            'spam_indicators': [
                'subscribe', 'check out', 'visit my', 'link in bio', 'click here',
                'follow me', 'dm me', 'first', 'early', 'notification squad',
                'www.', 'http://', 'https://', '.com', '.net', 'buy now', 'discount',
                'promo code', 'cheap', 'affordable', 'make money', 'earn cash'
            ]
        }
        
        # Category keywords for filtering
        self.category_keywords = {
            'skincare': ['skincare', 'acne', 'moisturizer', 'serum', 'cleanser', 'SPF', 'routine', 'skin', 'face', 'glow'],
            'fragrance': ['perfume', 'cologne', 'scent', 'fragrance', 'smell', 'notes', 'aroma', 'scented'],
            'makeup': ['makeup', 'foundation', 'lipstick', 'eyeshadow', 'mascara', 'concealer', 'cosmetics', 'blush'],
            'tech': ['technology', 'software', 'hardware', 'app', 'device', 'digital', 'computer', 'phone'],
            'gaming': ['game', 'gaming', 'player', 'console', 'PC', 'mobile gaming', 'level', 'character']
        }
        
        # Quality classification function
        self.classify_quality = lambda score: 'High' if score >= 3 else 'Medium' if score >= 1.5 else 'Low'

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
        if pd.isna(comment) or not isinstance(comment, str):
            return 0
        
        comment_lower = comment.lower().strip()
        if len(comment_lower) < 3:  # Minimum meaningful comment
            return 0
            
        score = 0
        
        # Length factor (more generous curve)
        length = len(comment_lower)
        if length >= 10:  # More reasonable minimum
            score += 1
        if 30 <= length <= 250:  # Optimal range is wider
            score += 2
        elif length > 250:  # Long but not necessarily bad
            score += 1
        
        # High quality keywords (broader matching)
        for word in self.quality_keywords['high_quality']:
            if word in comment_lower:
                score += 1.5  # Reduced from 2
                break  # Only count once per category
        
        # Engagement indicators
        engagement_words_found = 0
        for word in self.quality_keywords['engagement_words']:
            if word in comment_lower:
                engagement_words_found += 0.5  # Reduced from 1
                if engagement_words_found >= 2:  # Cap at 2 points
                    break
        
        score += min(engagement_words_found, 2)
        
        # Grammar and structure indicators
        if '?' in comment or '!' in comment:
            score += 0.5  # Reduced from 1
        
        if '.' in comment and len(comment.split('. ')) > 1:  # Multiple sentences
            score += 1
        
        # Penalty for spam indicators (less severe)
        spam_penalty = 0
        for spam_word in self.quality_keywords['spam_indicators']:
            if spam_word in comment_lower:
                spam_penalty += 1.5  # Reduced from 3
                if spam_penalty >= 3:  # Cap the penalty
                    break
        
        score -= spam_penalty
        
        # Penalty for excessive caps (more lenient)
        upper_count = sum(1 for char in comment if char.isupper())
        if upper_count > len(comment) * 0.4:  # 40% caps threshold instead of 30%
            score -= 1  # Reduced from 2
        
        # Penalty for repetitive characters (more lenient)
        if re.search(r'(.)\1{3,}', comment):  # 4+ repetitive chars instead of 2+
            score -= 0.5  # Reduced from 1
        
        # Bonus for good sentiment (NEW)
        try:
            blob = TextBlob(comment)
            if blob.sentiment.polarity > 0.3:  # Strongly positive
                score += 1
        except:
            pass
        
        return max(0, min(score, 10))  # Cap at 10, ensure non-negative

    def categorize_comment(self, comment):
        """Categorize comment based on keywords"""
        if pd.isna(comment):
            return ['Uncategorized']
        
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

    def debug_quality_score(self, comment):
        """Debug function to see why a comment gets a certain score"""
        if pd.isna(comment) or not isinstance(comment, str):
            return ["Invalid comment"]
        
        debug_info = []
        comment_lower = comment.lower()
        score = 0
        
        # Length analysis
        length = len(comment_lower)
        if length >= 10:
            debug_info.append(f"+1 for length ({length} chars)")
            score += 1
        if 30 <= length <= 250:
            debug_info.append(f"+2 for optimal length")
            score += 2
        elif length > 250:
            debug_info.append(f"+1 for long length")
            score += 1
        
        # Keyword analysis
        for word in self.quality_keywords['high_quality']:
            if word in comment_lower:
                debug_info.append(f"+1.5 for keyword: '{word}'")
                score += 1.5
                break
        
        # Engagement words
        engagement_words_found = 0
        for word in self.quality_keywords['engagement_words']:
            if word in comment_lower:
                engagement_words_found += 0.5
        if engagement_words_found > 0:
            debug_info.append(f"+{min(engagement_words_found, 2)} for engagement words")
            score += min(engagement_words_found, 2)
        
        # Grammar indicators
        if '?' in comment or '!' in comment:
            debug_info.append(f"+0.5 for question/exclamation")
            score += 0.5
        
        if '.' in comment and len(comment.split('. ')) > 1:
            debug_info.append(f"+1 for multiple sentences")
            score += 1
        
        debug_info.append(f"FINAL SCORE: {score} ({self.classify_quality(score)})")
        return debug_info

@st.cache_data
def load_data(uploaded_file):
    """Cache data loading to avoid reprocessing on every interaction"""
    if uploaded_file is None:
        return load_sample_data()
    
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return load_sample_data()

@st.cache_data
def analyze_data(_analyzer, df):
    """Cache the analysis results"""
    df = df.copy()
    df['sentiment'] = df['comment'].apply(_analyzer.analyze_sentiment)
    df['quality_score'] = df['comment'].apply(_analyzer.calculate_quality_score)
    df['is_spam'] = df['comment'].apply(_analyzer.detect_spam)
    df['categories'] = df['comment'].apply(_analyzer.categorize_comment)
    
    df['quality_category'] = df['quality_score'].apply(_analyzer.classify_quality)
    return df

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
        "Awesome video! Really helped me understand the product better.",
        "Good content, but could use more examples.",
        "I have a question about the application process?",
        "This worked perfectly for my skin type, thank you!",
        "Would you recommend this for oily skin?",
        "The scent is amazing, lasts all day!"
    ]
    
    video_ids = ['VID001', 'VID002', 'VID003', 'VID001', 'VID002', 'VID003', 'VID001', 'VID002', 'VID003', 'VID001', 'VID002', 'VID003', 'VID001', 'VID002', 'VID003', 'VID001', 'VID002', 'VID003', 'VID001', 'VID002']
    
    return pd.DataFrame({
        'video_id': video_ids,
        'comment': sample_comments,
        'likes': np.random.randint(0, 50, len(sample_comments)),
        'timestamp': pd.date_range('2024-01-01', periods=len(sample_comments), freq='H')
    })

def main():
    st.markdown('<h1 class="main-header">💬 CommentSense AI Analytics</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Comment Quality Analysis & Share of Engagement Metrics")
    
    # Initialize analyzer
    analyzer = CommentAnalyzer()
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Comment Dataset", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        # AUTOMATIC COLUMN MAPPING FOR DATATHON DATASET
        column_mapping = {}
        
        # Map comment text - MOST IMPORTANT
        if 'textOriginal' in df.columns:
            df = df.rename(columns={'textOriginal': 'comment'})
            column_mapping['textOriginal'] = 'comment'
            st.sidebar.success("✅ Detected 'textOriginal' column as comments")
        elif 'comment' not in df.columns:
            # Try to find text columns
            text_cols = [col for col in df.columns if 'text' in col.lower() or 'comment' in col.lower()]
            if text_cols:
                df = df.rename(columns={text_cols[0]: 'comment'})
                column_mapping[text_cols[0]] = 'comment'
                st.sidebar.info(f"📝 Using '{text_cols[0]}' column as comments")
            else:
                st.error("❌ Could not find comment column. Please ensure your dataset has a 'textOriginal' column or similar.")
                st.info("Falling back to sample data for demonstration.")
                df = load_sample_data()
        
        # Map video ID
        if 'videoId' in df.columns:
            df = df.rename(columns={'videoId': 'video_id'})
            column_mapping['videoId'] = 'video_id'
            st.sidebar.success("✅ Detected 'videoId' column")
        elif 'video_id' not in df.columns:
            vid_cols = [col for col in df.columns if 'video' in col.lower() and 'id' in col.lower()]
            if vid_cols:
                df = df.rename(columns={vid_cols[0]: 'video_id'})
                column_mapping[vid_cols[0]] = 'video_id'
                st.sidebar.info(f"🎬 Using '{vid_cols[0]}' column as video ID")
            else:
                df['video_id'] = 'VID001'  # Default value
                st.sidebar.warning("⚠️ No video ID column found. Using default value.")
        
        # Map like count
        if 'likeCount' in df.columns:
            df = df.rename(columns={'likeCount': 'likes'})
            column_mapping['likeCount'] = 'likes'
            st.sidebar.success("✅ Detected 'likeCount' column")
        elif 'likes' not in df.columns:
            like_cols = [col for col in df.columns if 'like' in col.lower() and 'count' in col.lower()]
            if like_cols:
                df = df.rename(columns={like_cols[0]: 'likes'})
                column_mapping[like_cols[0]] = 'likes'
                st.sidebar.info(f"👍 Using '{like_cols[0]}' column as likes")
            else:
                df['likes'] = 0  # Default value
                st.sidebar.warning("⚠️ No like count column found. Using default value (0).")
        
        # Map timestamp
        if 'publishedAt' in df.columns:
            df = df.rename(columns={'publishedAt': 'timestamp'})
            column_mapping['publishedAt'] = 'timestamp'
            # Convert to datetime if it's a string
            if df['timestamp'].dtype == 'object':
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            st.sidebar.success("✅ Detected 'publishedAt' column as timestamp")
        elif 'timestamp' not in df.columns:
            time_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'publish' in col.lower()]
            if time_cols:
                df = df.rename(columns={time_cols[0]: 'timestamp'})
                column_mapping[time_cols[0]] = 'timestamp'
                if df['timestamp'].dtype == 'object':
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                st.sidebar.info(f"⏰ Using '{time_cols[0]}' column as timestamp")
            else:
                df['timestamp'] = pd.Timestamp.now()  # Default value
                st.sidebar.warning("⚠️ No timestamp column found. Using current time.")
                
    else:
        st.info("📝 No file uploaded. Using sample data for demonstration.")
        df = load_sample_data()
        column_mapping = {}
    
    # Data preprocessing
    if 'comment' not in df.columns:
        st.error("Could not find a comment column in the dataset. Please check your file format
