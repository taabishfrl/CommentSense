from loaders import load_embedder, load_sentiment, load_zero_shot, load_toxicity
from utils import CAPS_RE, REPEAT_RE
import re

class CommentAnalyzer:
    def __init__(self):
        self.embedder = load_embedder()
        self.sentiment_pipe = load_sentiment()
        self.zs_pipe = load_zero_shot()
        self.tox_pipe = load_toxicity()

        self.quality_keywords = {
            "high_quality": [
                "insightful","helpful","informative","detailed","thoughtful",
                "constructive","in-depth","compare","recommend","results","review",
            ],
            "spam_indicators": [
                "subscribe","check out","visit my","link in bio","click here",
                "follow me","dm me","whatsapp","promo","discount","coupon",
            ],
        }
        self.categories = ["skincare","fragrance","makeup","hair","other"]

    def analyze_sentiment(self, comment: str) -> str:
        try:
            out = self.sentiment_pipe(str(comment))[0]["label"].upper()
            return {"NEGATIVE":"Negative","NEUTRAL":"Neutral","POSITIVE":"Positive"}.get(out,"Neutral")
        except Exception:
            return "Neutral"

    def detect_toxicity(self, comment: str) -> float:
        try:
            out = self.tox_pipe(str(comment), return_all_scores=True)[0]
            for item in out:
                if item["label"].lower() == "toxic":
                    return float(item["score"])
            return 0.0
        except Exception:
            return 0.0

    def zero_shot_categories(self, comment: str):
        try:
            out = self.zs_pipe(str(comment), candidate_labels=self.categories, multi_label=True)
            labs = [l for l, s in zip(out["labels"], out["scores"]) if s >= 0.40]
            return labs or ["other"]
        except Exception:
            return ["other"]

    def relevance_to_post(self, comment: str, post_text: str) -> float:
        if not isinstance(post_text, str) or not post_text.strip():
            return 0.0
        emb = self.embedder.encode([str(comment), str(post_text)], normalize_embeddings=True)
        sim = float((emb[0] * emb[1]).sum())
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))

    def quality_score(self, comment: str, post_text: str) -> float:
        if not isinstance(comment, str) or not comment.strip():
            return 0.0
        c = comment.lower()
        rel = self.relevance_to_post(comment, post_text)
        tox = self.detect_toxicity(comment)
        sent = self.analyze_sentiment(comment)
        sent_s = {"Positive": 1.0, "Neutral": 0.5, "Negative": 0.0}.get(sent, 0.5)
        length_bonus = 0.15 if 30 <= len(c) <= 220 else (0.05 if len(c) > 220 else 0.0)
        kw_bonus = 0.15 if any(k in c for k in self.quality_keywords["high_quality"]) else 0.0
        spam_pen = 0.25 if any(s in c for s in self.quality_keywords["spam_indicators"]) else 0.0
        caps_pen = 0.15 if len(CAPS_RE.findall(comment)) > max(1, len(comment)) * 0.35 else 0.0
        repeat_pen = 0.10 if REPEAT_RE.search(comment) else 0.0
        score = (0.45*rel + 0.25*sent_s + length_bonus + kw_bonus) - (0.35*tox + spam_pen + caps_pen + repeat_pen)
        return float(max(0.0, min(1.0, score)))
