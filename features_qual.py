import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans

# -----------------------
# Basic cleaning helpers
# -----------------------
STOP = set("""
a an the and or of for to in on with from this that is are be have has it its our your you we they i my me
""".split())

def _clean(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text).lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _token_ok(t: str) -> bool:
    if len(t) < 3: return False
    if t in STOP: return False
    if not re.search(r"[aeiou]", t): return False      # must have a vowel
    if re.search(r"(.)\1{2,}", t): return False        # aaaa, baaaad
    return True

# ----------------------------
# Existing (kept) theme helper
# ----------------------------
def _texts(df):
    return df["comment"].astype(str).tolist()

def cluster_themes(df, k=4):
    if len(df) < 4:
        return {}
    vect = TfidfVectorizer(stop_words="english", max_features=4000, ngram_range=(1,2))
    X = vect.fit_transform(_texts(df))
    k = min(k, len(df))
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    lab = km.fit_predict(X)
    feats = np.array(vect.get_feature_names_out())
    centers = km.cluster_centers_
    labels = {}
    for c in range(k):
        order = centers[c].argsort()[::-1][:6]
        key_terms = [feats[i] for i in order]
        labels[c] = key_terms
    df2 = df.copy()
    df2["theme_id"] = lab
    return {"labels": labels, "assignments": df2}

# ----------------------------------------
# Smarter "reasons for sentiment" (new)
# ----------------------------------------
THEME_MAP = [
    ("Performance / Speed", {"slow","lag","laggy","loading","load","delay","stuck","buffer","performance"}),
    ("Crashes & Bugs",      {"crash","freez","bug","glitch","error","broken","doesn","work","issue"}),
    ("Login / Account",     {"login","log in","sign in","password","auth","otp","account"}),
    ("Payments / Checkout", {"payment","pay","checkout","card","billing","invoice","charge","refund"}),
    ("Pricing / Fees",      {"price","expensive","cost","fee","subscription","paywall"}),
    ("Onboarding / UX",     {"confus","hard","difficult","complicated","onboard","tutorial","learn","discover"}),
    ("Notifications / Spam",{"notification","notif","spam","ads","advert","pop up"}),
    ("Content / Relevance", {"content","recommend","relevant","irrelevant","feed","search"}),
]

def _choose_min_df(n_docs: int) -> int:
    # adaptive min_df for small datasets
    if n_docs >= 50: return 3
    if n_docs >= 15: return 2
    return 1

def _extract_top_ngrams(docs: List[str], likes: Optional[np.ndarray], topk=20):
    if not docs:
        return []
    min_df = _choose_min_df(len(docs))
    for mdf in (min_df, 1):
        try:
            vec = CountVectorizer(
                ngram_range=(1,3),
                min_df=mdf,
                stop_words="english",
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # words only
            )
            X = vec.fit_transform(docs)
            if X.shape[1] == 0:
                continue
            vocab = np.array(vec.get_feature_names_out())
            # coverage per term weighted by likes (if provided)
            presence = (X > 0).astype(int)
            if likes is not None and len(likes) == X.shape[0]:
                w = (likes / (likes.max() if likes.max() > 0 else 1.0)) + 1.0  # 1..2
                weights = presence.T @ w
            else:
                weights = np.asarray(presence.sum(axis=0)).ravel()
            order = np.argsort(weights)[::-1][:topk]
            return [(vocab[i], float(weights[i])) for i in order]
        except ValueError:
            continue
    return []

def _map_to_theme(phrase: str) -> Optional[str]:
    p = phrase.lower()
    for label, keys in THEME_MAP:
        if any(k in p for k in keys):
            return label
    return None

def summarize_negative_reasons(df: pd.DataFrame, like_col: str = "likes", topk_themes: int = 3) -> List[Dict[str, Any]]:
    """
    Turn negative comments into business-readable themes with support stats.
    Returns: list of dicts: {theme, count, avg_likes, example}
    """
    neg = df[df["sentiment"] == "Negative"].copy()
    if neg.empty:
        return []

    # Prepare docs and likes
    docs = []
    idxs = []
    for i, row in neg.iterrows():
        s = _clean(str(row.get("comment", "")))
        toks = [t for t in s.split() if _token_ok(t)]
        if not toks: 
            continue
        docs.append(" ".join(toks))
        idxs.append(i)
    if not docs:
        return []

    likes = None
    if like_col in neg.columns:
        likes = neg.loc[idxs, like_col].fillna(0).astype(float).values
    top_ngrams = _extract_top_ngrams(docs, likes, topk=40)

    # Score themes by summed weights; keep an example comment
    theme_stats: Dict[str, Dict[str, Any]] = {}
    for (phrase, weight) in top_ngrams:
        label = _map_to_theme(phrase)
        if not label:
            continue
        if label not in theme_stats:
            theme_stats[label] = {"weight": 0.0, "rows": []}
        theme_stats[label]["weight"] += weight

    if not theme_stats:
        return []

    # For each theme, pick supporting rows and example
    results = []
    for label in theme_stats:
        # find rows that mention any theme keyword
        keys = next(k for (lab, k) in THEME_MAP if lab == label)
        mask = neg["comment"].str.lower().fillna("").apply(lambda s: any(k in s for k in keys))
        sub = neg[mask]
        if sub.empty:
            continue
        count = int(len(sub))
        avg_likes = float(sub[like_col].mean()) if like_col in sub.columns else 0.0
        # choose example = highest likes (or longest) snippet
        example = str(sub.loc[sub[like_col].idxmax(), "comment"])[:140] + "..." if like_col in sub.columns and len(sub) else str(sub.iloc[0]["comment"])[:140] + "..."
        results.append({
            "theme": label,
            "count": count,
            "avg_likes": round(avg_likes, 1),
            "example": example
        })

    # sort by count and cap
    results.sort(key=lambda x: (x["count"], x["avg_likes"]), reverse=True)
    return results[:topk_themes]

# -----------------------------------------
# (Legacy) unweighted reasons if you need it
# -----------------------------------------
def reasons_for_sentiment(df, topk=3):
    out = {}
    for name, filt in [("Positive", df[df["sentiment"]=="Positive"]),
                       ("Negative", df[df["sentiment"]=="Negative"])]:
        texts = filt["comment"].dropna().astype(str).tolist()
        if len(texts) < 2:
            out[name] = []
            continue
        for min_df in (2, 1):
            try:
                vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=min_df, max_df=0.95)
                X = vect.fit_transform(texts)
                if X.shape[1] == 0:
                    continue
                scores = np.asarray(X.sum(axis=0)).ravel()
                feats = np.array(vect.get_feature_names_out())
                order = np.argsort(scores)[::-1][:topk]
                out[name] = [feats[i] for i in order]
                break
            except ValueError:
                continue
        else:
            out[name] = []
    return out
