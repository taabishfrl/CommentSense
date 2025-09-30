import re
from collections import Counter
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# ------------------------
# Light text preprocessing
# ------------------------
STOP = set("""
a an the and or of for to in on with from this that is are be have has it its our your you we they i my me
""".split())

# Heuristics to avoid junk phrases in ads
BAN_WORDS = {
    "omg", "lol", "lmao", "idk", "wow", "nice", "pretty", "flag", "love", "pan", "man", "don",
    "thing", "stuff", "video", "content", "channel", "subscribe", "link", "bio"
}
# Hints that a phrase is descriptive/benefit-oriented
ADJ_HINTS = {
    "clean", "smooth", "fast", "simple", "easy", "intuitive", "beautiful", "sleek", "durable",
    "hydrating", "gentle", "lightweight", "natural", "long", "lasting", "reliable", "responsive",
    "comfortable", "quiet", "powerful", "compact", "precise", "affordable", "secure"
}


def _clean_tokens(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text).lower())
    toks = [t for t in text.split() if t not in STOP and len(t) > 2]
    return " ".join(toks)


# ---------------------------------------------
# Robust phrase extraction (resists empty vocab)
# ---------------------------------------------
def top_phrases(corpus: List[str], ngram=(1, 2), topk=10) -> List[str]:
    """
    Robust TF-IDF phrase extractor:
    - Cleans and drops empty docs
    - Tries min_df=2, then relaxes to min_df=1
    - Final fallback: simple frequency over tokens
    """
    cleaned = [_clean_tokens(x) for x in corpus if isinstance(x, str)]
    cleaned = [t for t in cleaned if t.strip()]
    if not cleaned:
        return []

    for min_df in (2, 1):
        try:
            vect = TfidfVectorizer(
                ngram_range=ngram,
                min_df=min_df,
                max_df=0.95,             # cap very common phrases
                stop_words="english",
            )
            X = vect.fit_transform(cleaned)
            if X.shape[1] == 0:
                continue
            scores = np.asarray(X.sum(axis=0)).ravel()
            feats = np.array(vect.get_feature_names_out())
            order = np.argsort(scores)[::-1][:topk]
            return [feats[i] for i in order]
        except ValueError:
            continue

    # Fallback: simple frequency on tokens
    bag = Counter()
    for t in cleaned:
        bag.update(t.split())
    return [w for w, _ in bag.most_common(topk)]


# ------------------------------------------------------
# Candidate selection and high-quality ad-line generator
# ------------------------------------------------------
def pick_positive_candidates(df, like_col: Optional[str] = "likes"):
    """
    Pick comments that are:
    - Positive sentiment
    - Not spam
    - Relevant (>= 0.5)
    - In top 30% by likes (if likes present)
    """
    if like_col not in df.columns:
        like_col = None

    base = df[
        (df["sentiment"] == "Positive")
        & (~df["is_spam"])
        & (df["relevance"] >= 0.5)
    ]
    if like_col and len(base):
        thr = base[like_col].quantile(0.7)
        base = base[base[like_col] >= thr]
    return base


def _is_meaningful_phrase(p: str) -> bool:
    toks = [t for t in re.findall(r"[a-zA-Z]{3,}", p.lower())]
    if len(toks) < 2:  # prefer bigrams+
        return False
    if any(t in BAN_WORDS for t in toks):
        return False
    # must have at least one descriptive hint or be a solid bigram+
    return any(t in ADJ_HINTS for t in toks) or len(toks) >= 2


def make_ad_slogans(positive_comments: List[str], topk: int = 3) -> List[str]:
    """
    Return up to `topk` concise, sensible ad claims.
    - Prefers meaningful bigrams and filters junk
    - Caps to 3 lines for neatness
    """
    phrases = top_phrases(positive_comments, ngram=(1, 2), topk=40)

    seen = set()
    cleaned = []
    for p in phrases:
        if " " not in p:  # skip unigrams like "love"
            continue
        base = re.sub(r"\s+", " ", p.lower().strip())
        if base in seen:
            continue
        if _is_meaningful_phrase(base):
            seen.add(base)
            cleaned.append(base)
        if len(cleaned) >= topk:
            break

    lines = [f"Loved for its {c}." for c in cleaned]
    if not lines:
        # final safe fallback
        lines = ["Loved by users for its clean, intuitive experience."]
    return lines[:topk]


# --------------------------------
# Creator outreach text template(s)
# --------------------------------
def outreach_template(creator_handle: Optional[str], product_name: str = "our new product") -> str:
    who = f"{creator_handle}" if creator_handle else "there"
    return (
        f"Hi {who}, we loved your recent video and the feedback from your audience. "
        f"Would you be open to a quick collab featuring {product_name}? "
        "We can provide a brief and assets. Keen to explore a fit!"
    )
