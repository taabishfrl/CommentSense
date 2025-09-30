import base64
import pathlib
import pandas as pd
import re
import streamlit as st

def logo_data_uri(path="assets/logo.png") -> str:
    p = pathlib.Path(path)
    if not p.exists():
        st.warning(f"Logo not found at {p.resolve()}")
        return ""
    return "data:image/" + p.suffix[1:] + ";base64," + base64.b64encode(p.read_bytes()).decode()

def detect_comment_column(df: pd.DataFrame):
    comment_priority = [
        'comment', 'text', 'textOriginal', 'content', 'message',
        'body', 'review', 'feedback', 'comment_text', 'comment_body',
        'user_comment', 'comment_message', 'post_comment', 'comment_content',
        'review_text', 'feedback_text', 'commentary', 'response'
    ]
    for col in comment_priority:
        if col in df.columns:
            return col
    for col in df.columns:
        col_lower = col.lower()
        if any(k in col_lower for k in ['comment','text','content','message','review','feedback','body']):
            return col
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            return col
    return df.columns[0] if len(df.columns) > 0 else None

def detect_numeric_column(df: pd.DataFrame, preferred_names):
    for col in preferred_names:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            return col
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None

def detect_date_column(df: pd.DataFrame):
    date_keywords = ['date', 'time', 'timestamp', 'created', 'published', 'posted', 'datetime']
    for col in date_keywords:
        if col in df.columns:
            return col
    for col in df.columns:
        if any(k in col.lower() for k in date_keywords):
            return col
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        try:
            pd.to_datetime(df[col].head(10))
            return col
        except Exception:
            continue
    return None

CAPS_RE = re.compile(r"[A-Z]")
REPEAT_RE = re.compile(r"(.)\\1{4,}")
