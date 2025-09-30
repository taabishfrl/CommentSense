import numpy as np
import pandas as pd

def load_sample_data():
    sample_comments = [
        "This tutorial was incredibly helpful! The step-by-step breakdown made it easy to follow.",
        "Great video! Could you do one about advanced techniques?",
        "First! Love your content!",
        "Really detailed explanation. Thank you!",
        "Subscribe to my channel for more content like this!",
        "I disagree with your approach, but I appreciate the thorough analysis.",
        "Wow!!! So good!!!",
        "Very informative. The examples were particularly useful.",
        "Can you do skincare routine for sensitive skin?",
        "Check out my latest video! Link in bio!",
        "This fragrance sounds interesting. What are the main notes?",
        "Your makeup tutorial inspired me to try new looks.",
        "The gaming review was spot on. Have you tried the new update?",
        "Thanks for the tech review. Very comprehensive.",
        "aaaaaaaaawesome video!!!!",
    ]
    video_ids = ["VID001","VID002","VID003"] * 5
    titles = {"VID001":"Hydrating Serum 101","VID002":"Summer Fragrance Picks","VID003":"Everyday Makeup Tips"}
    df = pd.DataFrame({
        "video_id": video_ids[:len(sample_comments)],
        "comment": sample_comments,
        "likes": np.random.randint(0, 50, len(sample_comments)),
        "shares": np.random.randint(0, 10, len(sample_comments)),
        "saves": np.random.randint(0, 10, len(sample_comments)),
        "timestamp": pd.date_range("2024-01-01", periods=len(sample_comments), freq="H"),
    })
    df["title"] = df["video_id"].map(titles)
    return df
