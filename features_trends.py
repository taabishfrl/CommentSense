import numpy as np
import pandas as pd

def daily_sentiment(df, date_col="timestamp"):
    if date_col not in df.columns:
        return pd.DataFrame()
    day = df.copy()
    day[date_col] = pd.to_datetime(day[date_col])
    day["day"] = day[date_col].dt.date
    agg = day.groupby("day").agg(
        total=("comment","count"),
        pos=("sentiment", lambda s: (s == "Positive").mean()),
        neg=("sentiment", lambda s: (s == "Negative").mean()),
        spam=("is_spam", "mean"),
        qcr=("quality_category", lambda s: (s == "High").mean())
    ).reset_index()
    return agg

def simple_forecast(series, horizon=7):
    # linear projection using last N points
    y = series.values.astype(float)
    if len(y) < 3:
        return np.array([y[-1] if len(y) else 0.0] * horizon)
    x = np.arange(len(y))
    a, b = np.polyfit(x, y, deg=1)
    future_x = np.arange(len(y), len(y)+horizon)
    pred = a*future_x + b
    return np.clip(pred, 0.0, 1.0)

def rising_terms(df, window=7):
    # crude trend on word frequency in last vs previous window across positives
    pos = df[df["sentiment"] == "Positive"]["comment"].astype(str).tolist()
    if len(pos) < 10:
        return []
    half = max(5, min(len(pos)//2, 100))
    old, new = pos[:-half], pos[-half:]
    def bag(lst):
        from collections import Counter
        import re
        cnt = Counter()
        for t in lst:
            for w in re.findall(r"[a-zA-Z]{3,}", t.lower()):
                cnt[w]+=1
        return cnt
    b_old, b_new = bag(old), bag(new)
    keys = set(b_new.keys()) | set(b_old.keys())
    gains = [(k, b_new.get(k,0) - b_old.get(k,0)) for k in keys]
    gains.sort(key=lambda x: x[1], reverse=True)
    return [k for k,v in gains[:10] if v>0]
