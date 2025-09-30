# features_wordcloud.py
# pip install wordcloud matplotlib (matplotlib is usually present with Streamlit/Plotly stacks)

from wordcloud import WordCloud
import matplotlib.pyplot as plt


def make_wordcloud(texts, width=1200, height=500, background_color="white"):
    wc = WordCloud(width=width, height=height, background_color=background_color)
    wc.generate(" ".join(texts))
    fig = plt.figure(figsize=(12, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    return fig
