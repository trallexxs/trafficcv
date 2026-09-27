"""Interactive Plotly figures for the website."""
from __future__ import annotations

from typing import List, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import CLASSES

PALETTE = {
    "accident": "#e5484d", "near_miss": "#f76b15", "red_light": "#ce2c31", "wrong_way": "#8e4ec6",
    "illegal_u_turn": "#ab4aba", "stopped_vehicle": "#ffc53d", "jaywalking": "#30a46c",
    "failure_to_yield": "#46a758", "illegal_turn": "#0090ff", "solid_line_crossing": "#12a594",
    "stop_line": "#3e63dd", "congestion": "#8b8d98", "road_obstacle": "#978365", "fire_smoke": "#1c2024",
}


def timeline_figure(events: Sequence[Sequence], duration: float, risk: List[List[float]] | None = None,
                    title: str = "") -> go.Figure:
    """Gantt-style event timeline (top) and Part B risk curve (bottom)."""
    labels = [c for c in CLASSES if any(e[2] == c for e in events)] or ["(no events)"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38],
                        vertical_spacing=0.06)
    for e in events:
        s, t, lab = float(e[0]), float(e[1]), e[2]
        fig.add_trace(go.Bar(
            x=[t - s], base=[s], y=[lab], orientation="h", marker_color=PALETTE.get(lab, "#555"),
            hovertemplate=f"<b>{lab}</b><br>{s:.1f} – {t:.1f} s ({t - s:.1f} s)<extra></extra>",
            showlegend=False), row=1, col=1)
    if risk:
        fig.add_trace(go.Scatter(x=[r[0] for r in risk], y=[r[1] for r in risk], mode="lines",
                                 line=dict(color="#e5484d", width=1.5), name="risk",
                                 hovertemplate="t=%{x:.1f}s risk=%{y:.2f}<extra></extra>"), row=2, col=1)
        fig.add_hline(y=0.5, line_dash="dot", line_color="#8b8d98", row=2, col=1)
    fig.update_yaxes(categoryorder="array", categoryarray=labels[::-1], row=1, col=1)
    fig.update_yaxes(range=[0, 1.02], title_text="P(accident ≤ 5 s)", row=2, col=1)
    fig.update_xaxes(range=[0, max(duration, 1.0)], title_text="time (s)", row=2, col=1)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40 if title else 10, b=10),
                      title=title or None, barmode="overlay", template="plotly_white",
                      hovermode="closest")
    return fig


def counts_figure(counts: dict, title: str) -> go.Figure:
    fig = go.Figure()
    for name in ("car", "truck", "bus", "motorcycle", "bicycle", "person"):
        if name in counts and sum(counts[name]):
            fig.add_trace(go.Scatter(x=counts["t"], y=counts[name], mode="lines", name=name, stackgroup=None))
    fig.update_layout(title=title, height=300, template="plotly_white", margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title="time (s)", yaxis_title="new tracks / 10 s")
    return fig


def bar_figure(x, y, title: str, xaxis: str, yaxis: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=x, y=y, marker_color="#0090ff"))
    fig.update_layout(title=title, height=300, template="plotly_white", margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title=xaxis, yaxis_title=yaxis)
    return fig
