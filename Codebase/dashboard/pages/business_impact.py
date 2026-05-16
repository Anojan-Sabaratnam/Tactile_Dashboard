import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output
import plotly.graph_objects as go
import plotly.express as px

dash.register_page(__name__, path="/business-impact", name="ROI & System Config")

# Risk Matrix Data
risk_data = [
    {"Risk": "Model Drift (New components)", "Impact": 4, "Probability": 3, "Category": "Technical", "Mitigation": "Continuous Learning Pipeline"},
    {"Risk": "Sensor Degradation", "Impact": 5, "Probability": 2, "Category": "Hardware", "Mitigation": "Auto-calibration routines"},
    {"Risk": "Network Disconnection", "Impact": 2, "Probability": 4, "Category": "Operational", "Mitigation": "Edge AI Fallback"},
    {"Risk": "False Positive Slips", "Impact": 3, "Probability": 3, "Category": "Performance", "Mitigation": "Confidence threshold tuning"},
    {"Risk": "Operator Alert Fatigue", "Impact": 4, "Probability": 2, "Category": "HMI", "Mitigation": "Smart alarm grouping"},
]

layout = html.Div(
    [
        html.Div(
            [
                html.H1("ROI & System Configuration", className="display-5 fw-bold text-gradient mb-2"),
                html.P(
                    "Configure factory parameters to project ROI and review system safeguards.",
                    className="lead text-secondary",
                ),
            ],
            className="mb-5 pb-3 border-bottom border-secondary border-opacity-25",
        ),
        
        # Row 1: ROI and Architecture
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Yield Optimization Calculator", className="fw-bold mb-4"),
                                
                                html.Label("Active Robots in Facility"),
                                dcc.Slider(10, 500, 10, value=50, id="calc-robots", className="mb-4", marks={10: '10', 250: '250', 500: '500'}),
                                
                                html.Label("Average Picks/Hr per Robot"),
                                dcc.Slider(100, 5000, 100, value=1000, id="calc-speed", className="mb-4", marks={100: '100', 5000: '5k'}),
                                
                                html.Label("Historical Defect Rate (%)"),
                                dcc.Slider(0.1, 5.0, 0.1, value=1.5, id="calc-defect", className="mb-4", marks={0.1: '0.1%', 5.0: '5.0%'}),
                                
                                html.Hr(className="border-secondary opacity-25 my-4"),
                                
                                html.H5("Projected Annual Savings", className="text-secondary mb-3 text-center"),
                                html.Div(id="roi-result", className="display-4 fw-bold text-success text-center mb-2"),
                                html.P("Based on active Edge AI stability prediction metrics.", className="text-center text-muted small")
                            ]
                        ),
                        className="glass-panel border-0 h-100",
                    ),
                    width=12, lg=5, className="mb-4"
                ),
                
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Network Architecture Telemetry", className="fw-bold mb-4"),
                                    html.Div(
                                        [
                                            html.Div([
                                                html.H6("Central Cloud Mode (Disabled)", className="text-danger mb-2"),
                                                html.Div(dbc.Progress(value=95, color="danger", className="mb-1", style={"height": "8px"})),
                                                html.Span("High Latency (>50ms) & Bandwidth (43 Mbps)", className="small text-muted")
                                            ], className="mb-4"),
                                            
                                            html.Div([
                                                html.H6("Distributed Edge Mode (Active)", className="text-success mb-2 text-gradient"),
                                                html.Div(dbc.Progress(value=15, color="success", className="mb-1", style={"height": "8px"})),
                                                html.Span("Ultra-low Latency (<10ms) & Bandwidth (3.7 Mbps)", className="small text-muted")
                                            ])
                                        ]
                                    ),
                                    html.P("Local processing prevents network bottlenecks and ensures real-time reactions.", className="text-muted small mt-4")
                                ]
                            ),
                            className="glass-panel border-0 h-100",
                        ),
                    ],
                    width=12, lg=7, className="mb-4"
                )
            ]
        ),
        
        # Row 2: Ethics and Risk
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Active Threat & Mitigation Matrix", className="fw-bold mb-3"),
                        dcc.Graph(
                            figure=px.scatter(
                                risk_data, x="Probability", y="Impact", color="Category", 
                                text="Risk", hover_data=["Mitigation"],
                                size=[1]*len(risk_data), size_max=15
                            ).update_traces(
                                textposition="top center", textfont=dict(color='white')
                            ).update_layout(
                                template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                margin=dict(l=20, r=20, t=20, b=20),
                                xaxis=dict(range=[0.5, 5.5], title="Probability Index"),
                                yaxis=dict(range=[0.5, 5.5], title="Severity Index")
                            ),
                            config={'displayModeBar': False}, style={"height": "300px"}
                        )
                    ]),
                    className="glass-panel border-0 h-100"
                ),
                width=12, lg=7, className="mb-4"
            ),
            
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Workforce & Safety Safeguards", className="fw-bold mb-4 text-warning"),
                        html.Div([
                            html.H6([html.I(className="bi bi-person-check-fill me-2"), "Operator Augmentation"], className="fw-bold"),
                            html.P("System alerts human supervisors only on anomalous drop events, reducing cognitive load.", className="small text-secondary mb-4"),
                            
                            html.H6([html.I(className="bi bi-shield-lock-fill me-2"), "HRC Safety Assurance"], className="fw-bold"),
                            html.P("Automatic sub-10ms reflex triggers prevent robotic arms from applying dangerous excessive force near personnel.", className="small text-secondary mb-4"),
                            
                            html.H6([html.I(className="bi bi-diagram-3-fill me-2"), "Material Handling Equivalence"], className="fw-bold"),
                            html.P("Calibration profiles loaded for 51 distinct materials to prevent handling bias.", className="small text-secondary")
                        ])
                    ]),
                    className="glass-panel border-0 h-100"
                ),
                width=12, lg=5, className="mb-4"
            )
        ])
    ],
    className="animate__animated animate__fadeIn",
)

@callback(
    Output("roi-result", "children"),
    [Input("calc-robots", "value"),
     Input("calc-speed", "value"),
     Input("calc-defect", "value")]
)
def update_roi(robots, speed, defect_rate):
    annual_components = robots * speed * 16 * 250
    current_drops = annual_components * (defect_rate / 100)
    prevented_drops = current_drops * 0.74
    savings = prevented_drops * 0.15
    
    if savings > 1_000_000:
        return f"${savings/1_000_000:.1f}M"
    else:
        return f"${savings:,.0f}"
