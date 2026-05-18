import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

dash.register_page(__name__, path="/historical-analytics", name="Historical Analytics")

# Generate dummy historical data for 30 days (seeded so charts are stable across restarts)
_rng = np.random.default_rng(seed=42)
dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
slips_prevented = _rng.normal(150, 20, 30).astype(int)
slips_prevented[15] = 45   # simulate a day where line was down
slips_prevented[28] = 210  # simulate a high throughput day

efficiency = np.clip(_rng.normal(98.5, 0.5, 30), 90, 100)
efficiency[15] = 92.1

fig_slips = go.Figure(data=[
    go.Bar(
        x=dates, y=slips_prevented,
        marker_color='#00e5ff',
        name='Slips Prevented'
    )
])
fig_slips.update_layout(
    title="Daily Slips Prevented (Last 30 Days)",
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=50, b=20),
    height=300
)

fig_eff = go.Figure(data=[
    go.Scatter(
        x=dates, y=efficiency,
        mode='lines+markers',
        line=dict(color='#00e676', width=3),
        name='Overall Yield Efficiency (%)'
    )
])
fig_eff.update_layout(
    title="Production Yield Efficiency",
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=50, b=20),
    yaxis=dict(range=[90, 100]),
    height=300
)


layout = html.Div(
    [
        html.Div(
            [
                html.H1("Historical Analytics", className="display-5 fw-bold text-gradient mb-2"),
                html.P(
                    "Review long-term trends in production yield and AI system interventions.",
                    className="lead text-secondary",
                ),
            ],
            className="mb-4 pb-3 border-bottom border-secondary border-opacity-25",
        ),
        
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([dcc.Graph(figure=fig_slips, config={'displayModeBar': False})]),
                        className="glass-panel border-0 mb-4",
                    ), width=12, lg=6
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([dcc.Graph(figure=fig_eff, config={'displayModeBar': False})]),
                        className="glass-panel border-0 mb-4",
                    ), width=12, lg=6
                )
            ]
        ),
        
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("Monthly Performance Summary", className="fw-bold mb-4"),
                            html.Div([
                                html.H6("Total Components Handled", className="text-secondary mb-1"),
                                html.H3("1,452,800", className="text-light fw-bold mb-4"),
                                
                                html.H6("Total Slips Prevented by AI", className="text-secondary mb-1"),
                                html.H3("4,420", className="text-info fw-bold mb-4"),
                                
                                html.H6("Estimated Material Savings", className="text-secondary mb-1"),
                                html.H3("$663.00", className="text-success fw-bold mb-4"),
                                
                                html.H6("Average Sensor Uptime", className="text-secondary mb-1"),
                                html.H3("99.8%", className="text-light fw-bold"),
                            ])
                        ]),
                        className="glass-panel border-0 h-100",
                    ), width=12
                )
            ]
        )
    ],
    className="animate__animated animate__fadeIn",
)
