import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output
import plotly.express as px

dash.register_page(__name__, path="/business-impact", name="ROI & System Config")

# ── ROI constants ─────────────────────────────────────────────────────────────
_SHIFTS_PER_DAY = 2
_WORKING_DAYS_PER_YEAR = 250
# 22.9pp = RF accuracy (74.3%) minus no-AI baseline (51.3%) — Garcia-Garcia et al.
_AI_IMPROVEMENT = 0.229

# Risk Matrix Data
risk_data = [
    {"Risk": "Model Drift (New components)", "Impact": 4, "Probability": 3, "Category": "Technical",   "Mitigation": "Continuous Learning Pipeline"},
    {"Risk": "Sensor Degradation",           "Impact": 5, "Probability": 2, "Category": "Hardware",    "Mitigation": "Auto-calibration routines"},
    {"Risk": "Network Disconnection",        "Impact": 2, "Probability": 4, "Category": "Operational", "Mitigation": "Edge AI Fallback"},
    {"Risk": "False Positive Slips",         "Impact": 3, "Probability": 3, "Category": "Performance", "Mitigation": "Confidence threshold tuning"},
    {"Risk": "Operator Alert Fatigue",       "Impact": 4, "Probability": 2, "Category": "HMI",         "Mitigation": "Smart alarm grouping"},
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

        # ── Row 1: ROI Calculator & Network Architecture ───────────────────────
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Electronics Line ROI Calculator", className="fw-bold mb-1"),
                        html.P(
                            "Figures anchored to IPC-A-610 defect benchmarks and SMT industry data.",
                            className="text-muted small mb-4",
                        ),

                        html.Label("Components Handled per Shift"),
                        dcc.Slider(
                            1000, 50000, 1000, value=12000, id="calc-components",
                            className="mb-4",
                            marks={1000: "1k", 12000: "12k", 25000: "25k", 50000: "50k"},
                        ),

                        html.Label("Avg Component Value (£)"),
                        html.P(
                            className="text-muted small mb-1",
                        ),
                        dcc.Slider(
                            1, 100, 1, value=15, id="calc-value",
                            className="mb-4",
                            marks={1: "£1", 15: "£15", 50: "£50", 100: "£100"},
                        ),

                        html.Label("Slip Rate Without AI (%)"),
                        html.P(
                            className="text-muted small mb-1",
                        ),
                        dcc.Slider(
                            0.5, 5.0, 0.1, value=2.1, id="calc-slip",
                            className="mb-4",
                            marks={0.5: "0.5%", 2.1: "2.1%", 5.0: "5.0%"},
                        ),

                        html.Label("Edge AI Deployment Cost (£)"),
                        html.P(
                            className="text-muted small mb-1",
                        ),
                        dcc.Slider(
                            10000, 100000, 5000, value=25000, id="calc-deploy",
                            className="mb-4",
                            marks={10000: "£10k", 25000: "£25k", 55000: "£55k", 100000: "£100k"},
                        ),

                        html.Hr(className="border-secondary opacity-25 my-3"),

                        # Results
                        dbc.Row([
                            dbc.Col([
                                html.P("Annual Savings", className="text-secondary small text-center mb-1"),
                                html.Div(id="roi-result", className="display-5 fw-bold text-success text-center"),
                            ], width=6),
                            dbc.Col([
                                html.P("Payback Period", className="text-secondary small text-center mb-1"),
                                html.Div(id="roi-payback", className="display-5 fw-bold text-info text-center"),
                            ], width=6),
                        ], className="mb-3"),
                    ]),
                    className="glass-panel border-0 h-100",
                ),
                width=12, lg=5, className="mb-4",
            ),

            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Network Architecture Telemetry", className="fw-bold mb-4"),
                        html.Div([
                            html.Div([
                                html.H6("Central Cloud Mode (Disabled)", className="text-danger mb-2"),
                                dbc.Progress(value=95, color="danger", className="mb-1", style={"height": "8px"}),
                                html.Span("High Latency (>50ms) & Bandwidth (43 Mbps)", className="small text-muted"),
                            ], className="mb-4"),
                            html.Div([
                                html.H6("Distributed Edge Mode (Active)", className="text-success mb-2 text-gradient"),
                                dbc.Progress(value=15, color="success", className="mb-1", style={"height": "8px"}),
                                html.Span("Ultra-low Latency (<10ms) & Bandwidth (3.7 Mbps)", className="small text-muted"),
                            ]),
                        ]),
                        html.P(
                            "Local processing prevents network bottlenecks and ensures real-time reactions.",
                            className="text-muted small mt-4",
                        ),
                    ]),
                    className="glass-panel border-0 h-100",
                ),
            ], width=12, lg=7, className="mb-4"),
        ]),

        # ── Row 2: Risk Matrix & Safety Safeguards ────────────────────────────
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Active Threat & Mitigation Matrix", className="fw-bold mb-3"),
                        dcc.Graph(
                            figure=px.scatter(
                                risk_data, x="Probability", y="Impact", color="Category",
                                text="Risk", hover_data=["Mitigation"],
                                size=[1] * len(risk_data), size_max=15,
                            ).update_traces(
                                textposition="top center", textfont=dict(color="white"),
                            ).update_layout(
                                template="plotly_dark",
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                margin=dict(l=20, r=20, t=20, b=20),
                                xaxis=dict(range=[0.5, 5.5], title="Probability Index"),
                                yaxis=dict(range=[0.5, 5.5], title="Severity Index"),
                            ),
                            config={"displayModeBar": False},
                            style={"height": "300px"},
                        ),
                    ]),
                    className="glass-panel border-0 h-100",
                ),
                width=12, lg=7, className="mb-4",
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
                            html.P("Calibration profiles loaded for 51 distinct materials to prevent handling bias.", className="small text-secondary"),
                        ]),
                    ]),
                    className="glass-panel border-0 h-100",
                ),
                width=12, lg=5, className="mb-4",
            ),
        ]),
    ],
    className="animate__animated animate__fadeIn",
)


@callback(
    [Output("roi-result", "children"),
     Output("roi-payback", "children")],
    [Input("calc-components", "value"),
     Input("calc-value", "value"),
     Input("calc-slip", "value"),
     Input("calc-deploy", "value")],
)
def update_roi(components, component_value, slip_rate, deploy_cost):
    annual_components = components * _SHIFTS_PER_DAY * _WORKING_DAYS_PER_YEAR
    slips_without_ai = annual_components * (slip_rate / 100)
    slips_caught = slips_without_ai * _AI_IMPROVEMENT
    annual_savings = slips_caught * component_value

    if annual_savings >= 1_000_000:
        savings_str = f"£{annual_savings / 1_000_000:.1f}M"
    elif annual_savings >= 1_000:
        savings_str = f"£{annual_savings:,.0f}"
    else:
        savings_str = f"£{annual_savings:.0f}"

    if annual_savings > 0:
        payback_months = (deploy_cost / annual_savings) * 12
        if payback_months < 1:
            payback_str = "<1 mo"
        elif payback_months > 120:
            payback_str = ">10 yrs"
        else:
            payback_str = f"{payback_months:.1f} mo"
    else:
        payback_str = "N/A"

    return savings_str, payback_str
