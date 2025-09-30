import plotly.graph_objects as go

def style_plotly_chart(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        margin=dict(t=50, r=20, b=40, l=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(font=dict(color="white")),
    )
    fig.update_xaxes(color="white", gridcolor="rgba(255,255,255,0.2)")
    fig.update_yaxes(color="white", gridcolor="rgba(255,255,255,0.2)")
    if fig.layout.annotations:
        for a in fig.layout.annotations:
            a.font = a.font or dict()
            a.font.color = "white"
    return fig
