from textblob import TextBlob
import pandas as pd
import re

class CommentAnalyzer:
    def __init__(self):
        # Define quality indicators
        self.quality_keywords = {
            'high_quality': [
                'insightful', 'helpful', 'informative', 'detailed', 'thoughtful', 
                'well-explained', 'comprehensive', 'valuable', 'educational', 'clear'
            ],
            'engagement_words': [
                'question', 'why', 'how', 'what', 'when', 'where', 'discuss',
                'thoughts', 'opinion', 'agree', 'disagree', 'interesting'
            ],
            'spam_indicators': [
                'subscribe', 'check out', 'visit my', 'link in bio', 'click here',
                'follow me', 'dm me', 'first', 'early', 'notification squad'
            ]
        }
        
        self.category_keywords = {
            'skincare': ['skincare', 'acne', 'moisturizer', 'serum', 'cleanser', 'SPF'],
            'makeup': ['makeup', 'foundation', 'lipstick', 'eyeshadow', 'mascara'],
            'tech': ['technology', 'software', 'hardware', 'app', 'device', 'digital']
        }

    def analyze_comment(self, comment):
        if pd.isna(comment):
            return {
                'sentiment': 'Neutral',
                'quality_score': 0,
                'quality_category': 'Low',
                'is_spam': False,
                'categories': ['Uncategorized']
            }
            
        text = str(comment).lower()
        
        # Calculate scores
        quality_score = self.calculate_quality_score(text)
        sentiment = self.analyze_sentiment(text)
        categories = self.categorize_comment(text)
        is_spam = self.detect_spam(text)
        
        # Determine quality category
        if quality_score >= 4:
            quality_category = 'High'
        elif quality_score >= 2:
            quality_category = 'Medium'
        else:
            quality_category = 'Low'
            
        return {
            'sentiment': sentiment,
            'quality_score': quality_score,
            'quality_category': quality_category,
            'is_spam': is_spam,
            'categories': categories
        }

    def analyze_sentiment(self, comment):
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
        if pd.isna(comment):
            return 0
        
        comment_lower = str(comment).lower()
        score = 0
        
        length = len(comment_lower)
        if 20 <= length <= 200:
            score += 2
        elif length > 200:
            score += 1
        
        for word in self.quality_keywords['high_quality']:
            if word in comment_lower:
                score += 2
        
        for word in self.quality_keywords['engagement_words']:
            if word in comment_lower:
                score += 1
        
        if '?' in comment or '!' in comment:
            score += 1
        
        for spam_word in self.quality_keywords['spam_indicators']:
            if spam_word in comment_lower:
                score -= 3
        
        if len(re.findall(r'[A-Z]', comment)) > len(comment) * 0.3:
            score -= 2
        
        if re.search(r'(.)\1{2,}', comment):
            score -= 1
        
        return max(0, score)

    def categorize_comment(self, comment):
        if pd.isna(comment):
            return 'Uncategorized'
        
        comment_lower = str(comment).lower()
        categories = []
        
        for category, keywords in self.category_keywords.items():
            if any(keyword in comment_lower for keyword in keywords):
                categories.append(category)
        
        return categories if categories else ['Uncategorized']

    def detect_spam(self, comment):
        if pd.isna(comment):
            return False
        
        comment_lower = str(comment).lower()
        spam_count = sum(1 for spam_word in self.quality_keywords['spam_indicators'] 
                        if spam_word in comment_lower)
        
        if len(comment_lower) < 5:
            return True
        if spam_count >= 2:
            return True
        if re.search(r'(.)\1{4,}', comment):
            return True
        
        return False