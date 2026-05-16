import json
from pathlib import Path
import random
from collections import deque
import numpy as np

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output, State

dash.register_page(__name__, path="/live-demo", name="Live Demo")

# Simulated Data Queues
MAX_POINTS = 50
time_queue = deque(maxlen=MAX_POINTS)
taxel_mean_queue = deque(maxlen=MAX_POINTS)
prob_rf_queue = deque(maxlen=MAX_POINTS)
prob_cnn_queue = deque(maxlen=MAX_POINTS)

# Initialize queues
for i in range(MAX_POINTS):
    time_queue.append(i)
    taxel_mean_queue.append(0)
    prob_rf_queue.append(0)
    prob_cnn_queue.append(0)

layout = html.Div(
    [
        html.Div(
            [
                html.H1("Live Edge Inference", className="display-5 fw-bold text-gradient mb-2"),
                html.P(
                    "Real-time streaming simulation of 74-channel tactile data and edge AI predictions.",
                    className="lead text-secondary",
                ),
            ],
            className="mb-4 pb-3 border-bottom border-secondary border-opacity-25",
        ),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Stream Control", className="fw-bold bg-transparent border-0"),
                    dbc.CardBody([
                        dbc.ButtonGroup([
                            dbc.Button([html.I(className="bi bi-play-fill me-2"), "Start Stream"], id="btn-start", color="success", outline=True, className="fw-bold"),
                            dbc.Button([html.I(className="bi bi-stop-fill me-2"), "Stop Stream"], id="btn-stop", color="danger", outline=True, className="fw-bold"),
                        ], className="w-100 mb-4"),
                        
                        html.Div(id="slip-alert-box", className="p-3 text-center rounded-3 border border-secondary border-opacity-25 bg-darker mt-3 transition-all"),
                        
                        html.Hr(className="border-secondary opacity-25 my-4"),
                        html.H5("Multi-Sensory Fusion", className="text-secondary mb-3"),
                        dcc.Graph(id="taxel-heatmap", config={'displayModeBar': False}, style={"height": "200px"}),
                    ])
                ], className="glass-panel border-0 h-100")
            ], width=12, lg=4, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Time-Series Inference", className="fw-bold bg-transparent border-0"),
                    dbc.CardBody([
                        dcc.Graph(id="live-timeseries", config={'displayModeBar': False}, style={"height": "400px"}),
                    ])
                ], className="glass-panel border-0 h-100")
            ], width=12, lg=8, className="mb-4")
        ]),
        
        dcc.Interval(id="stream-interval", interval=200, n_intervals=0, disabled=True),
        dcc.Store(id="sim-state", data={"is_slipping": False, "slip_ticks_left": 0, "tick_count": 0})
    ],
    className="animate__animated animate__fadeIn",
)

@callback(
    Output("stream-interval", "disabled"),
    [Input("btn-start", "n_clicks"), Input("btn-stop", "n_clicks")],
    prevent_initial_call=True
)
def toggle_stream(start, stop):
    ctx = dash.callback_context
    if not ctx.triggered:
        return True
    btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if btn_id == "btn-start":
        time_queue.clear()
        taxel_mean_queue.clear()
        prob_rf_queue.clear()
        prob_cnn_queue.clear()
        for i in range(MAX_POINTS):
            time_queue.append(i)
            taxel_mean_queue.append(0)
            prob_rf_queue.append(0)
            prob_cnn_queue.append(0)
        return False
        
    return True

@callback(
    [Output("live-timeseries", "figure"),
     Output("taxel-heatmap", "figure"),
     Output("slip-alert-box", "children"),
     Output("slip-alert-box", "className"),
     Output("sim-state", "data")],
    Input("stream-interval", "n_intervals"),
    State("sim-state", "data"),
    prevent_initial_call=True
)
def update_stream(n, state):
    tick = state["tick_count"] + 1
    
    # Randomly trigger a slip event (about every 30 ticks)
    if not state["is_slipping"] and random.random() < 0.03:
        state["is_slipping"] = True
        state["slip_ticks_left"] = random.randint(5, 12)
        
    if state["is_slipping"]:
        taxel_base = random.uniform(0.6, 1.0)
        prob_rf = min(1.0, random.uniform(0.8, 1.0))
        prob_cnn = min(1.0, random.uniform(0.75, 0.95))
        state["slip_ticks_left"] -= 1
        if state["slip_ticks_left"] <= 0:
            state["is_slipping"] = False
    else:
        taxel_base = random.uniform(0.0, 0.3)
        prob_rf = random.uniform(0.0, 0.2)
        prob_cnn = random.uniform(0.0, 0.25)
        
    time_queue.append(tick)
    taxel_mean_queue.append(taxel_base)
    prob_rf_queue.append(prob_rf)
    prob_cnn_queue.append(prob_cnn)
    
    # Update Time-Series
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=list(time_queue), y=list(taxel_mean_queue),
        mode='lines', name='Avg Taxel Pressure',
        line=dict(color='rgba(255, 255, 255, 0.3)', width=2),
        fill='tozeroy', fillcolor='rgba(255, 255, 255, 0.05)'
    ))
    fig_ts.add_trace(go.Scatter(
        x=list(time_queue), y=list(prob_rf_queue),
        mode='lines', name='RF Slip Prob',
        line=dict(color='#00e5ff', width=3)
    ))
    fig_ts.add_trace(go.Scatter(
        x=list(time_queue), y=list(prob_cnn_queue),
        mode='lines', name='CNN Slip Prob',
        line=dict(color='#b000ff', width=3, dash='dot')
    ))
    
    fig_ts.add_hline(y=0.5, line_dash="dash", line_color="rgba(255, 23, 68, 0.5)", annotation_text="Slip Threshold")
    
    fig_ts.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, 1.1], gridcolor='rgba(255,255,255,0.1)', zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update Heatmap (3 fingers x 24 taxels)
    heatmap_data = np.random.normal(taxel_base, 0.1, (3, 24))
    heatmap_data = np.clip(heatmap_data, 0, 1)
    
    fig_hm = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        colorscale='Viridis',
        showscale=False
    ))
    fig_hm.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, ticktext=['F1', 'F2', 'F3'], tickvals=[0, 1, 2]),
    )
    
    # Update Alert Box
    if state["is_slipping"]:
        alert_ui = [
            html.H3("SLIP DETECTED", className="text-danger fw-bold mb-1 font-mono"),
            html.P("Triggering emergency grasp reflex...", className="text-danger small mb-0")
        ]
        alert_class = "p-3 text-center rounded-3 border border-danger bg-darker mt-3 transition-all alert-slip"
    else:
        alert_ui = [
            html.H4("STABLE GRASP", className="text-success fw-bold mb-1 font-mono"),
            html.P("Monitoring taxel array...", className="text-secondary small mb-0")
        ]
        alert_class = "p-3 text-center rounded-3 border border-success border-opacity-25 bg-darker mt-3 transition-all"
        
    state["tick_count"] = tick
    
    return fig_ts, fig_hm, alert_ui, alert_class, state
