import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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
        "Very informative video. The examples you provided were particularly useful.",
        "Can you please make a video about skincare routine for sensitive skin?",
        "Check out my latest video! Link in bio!",
        "This fragrance sounds interesting. What are the main notes?",
        "Your makeup tutorial always inspire me to try new looks.",
        "The gaming review was spot on. Have you tried the new update?",
        "Thanks for the tech review. Very comprehensive analysis.",
        "aaaaaaaaawesome video!!!!"
    ]
    
    # Make sure all arrays have the same length as sample_comments
    n_samples = len(sample_comments)
    
    # Create repeating video IDs
    video_ids = ['VID001', 'VID002', 'VID003'] * ((n_samples + 2) // 3)
    video_ids = video_ids[:n_samples]  # Trim to match number of comments
    
    return pd.DataFrame({
        'video_id': video_ids,
        'comment': sample_comments,
        'likes': np.random.randint(0, 50, n_samples),
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H')
    })