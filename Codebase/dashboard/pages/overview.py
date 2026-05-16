import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

dash.register_page(__name__, path="/", name="Fleet Overview")

def create_kpi_card(title, value, subtitle, icon, color):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.I(className=f"bi {icon} fs-1 text-{color} opacity-75"),
                        html.Div(
                            [
                                html.H6(title, className="text-uppercase text-muted fw-bold mb-1"),
                                html.Div(value, className=f"metric-value text-{color}"),
                                html.Small(subtitle, className="text-secondary"),
                            ],
                            className="text-end",
                        ),
                    ],
                    className="d-flex justify-content-between align-items-center",
                )
            ]
        ),
        className="glass-panel border-0 h-100",
    )

layout = html.Div(
    [
        html.Div(
            [
                html.H1("Command Center", className="display-5 fw-bold text-gradient mb-2"),
                html.P(
                    "Real-time monitoring of AI-enhanced tactile sensors across the production floor.",
                    className="lead text-secondary",
                ),
            ],
            className="mb-5 pb-3 border-bottom border-secondary border-opacity-25",
        ),
        
        # KPI Row
        dbc.Row(
            [
                dbc.Col(
                    create_kpi_card(
                        "Active Sensors", "124/130", "6 offline for maintenance", "bi-cpu", "primary"
                    ),
                    width=12, md=6, xl=3, className="mb-4",
                ),
                dbc.Col(
                    create_kpi_card(
                        "Slips Prevented", "4,291", "Last 24 Hours", "bi-shield-check", "success"
                    ),
                    width=12, md=6, xl=3, className="mb-4",
                ),
                dbc.Col(
                    create_kpi_card(
                        "Edge Latency", "8.4ms", "Average inference time", "bi-lightning-charge", "warning"
                    ),
                    width=12, md=6, xl=3, className="mb-4",
                ),
                dbc.Col(
                    create_kpi_card(
                        "Network Load", "3.7 Mbps", "Optimized via Edge AI", "bi-speedometer2", "info"
                    ),
                    width=12, md=6, xl=3, className="mb-4",
                ),
            ]
        ),
        
        # Content Row
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("System Status", className="fw-bold mb-3"),
                                    html.P(
                                        "The Edge AI tactile control system is currently running optimally. All active robot arms "
                                        "are processing grasp stability locally, ensuring immediate reaction to slip events "
                                        "without relying on cloud server latency."
                                    ),
                                    html.H5("Active Production Lines", className="fw-bold mb-3 mt-4 text-secondary"),
                                    html.Div([
                                        html.Div([html.Span("Line A - SMT Assembly"), dbc.Progress(value=98, color="success", className="mb-3", style={"height": "10px"})]),
                                        html.Div([html.Span("Line B - Heavy Payload"), dbc.Progress(value=100, color="success", className="mb-3", style={"height": "10px"})]),
                                        html.Div([html.Span("Line C - Inspection"), dbc.Progress(value=45, color="warning", className="mb-3", style={"height": "10px"})]),
                                    ])
                                ]
                            ),
                            className="glass-panel border-0 h-100",
                        )
                    ],
                    width=12,
                    lg=6,
                    className="mb-4",
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Sensor Blueprint", className="fw-bold mb-3 text-center"),
                                    html.Img(
                                        src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/BioTac_Sensor.jpg/800px-BioTac_Sensor.jpg",
                                        className="img-fluid rounded-3 opacity-75 mb-3",
                                        style={"filter": "brightness(0.8) contrast(1.2) grayscale(0.5)"}
                                    ),
                                    html.P(
                                        "BioTac SP 3-Finger array: 74 distinct taxels monitored continuously.",
                                        className="text-center text-muted small"
                                    ),
                                ]
                            ),
                            className="glass-panel border-0 h-100 d-flex flex-column justify-content-center",
                        )
                    ],
                    width=12,
                    lg=6,
                    className="mb-4",
                ),
            ]
        ),
    ],
    className="animate__animated animate__fadeIn",
)
