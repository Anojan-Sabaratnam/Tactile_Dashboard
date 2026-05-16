import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
import random

dash.register_page(__name__, path="/fleet-status", name="Robot Fleet Status")

def generate_robot_card(robot_id, status, line, battery, uptime):
    if status == "Active":
        color = "success"
        icon = "bi-check-circle-fill"
        status_text = "Online & Processing"
    elif status == "Calibrating":
        color = "warning"
        icon = "bi-arrow-repeat"
        status_text = "Sensor Calibration"
    else:
        color = "danger"
        icon = "bi-exclamation-triangle-fill"
        status_text = "Offline / Maintenance"
        
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.H5(f"RoboArm #{robot_id}", className="fw-bold mb-0"),
                    html.I(className=f"bi {icon} text-{color} fs-4")
                ], className="d-flex justify-content-between align-items-center mb-3"),
                
                html.P([html.Strong("Line: "), line], className="small mb-1 text-secondary"),
                html.P([html.Strong("Status: "), html.Span(status_text, className=f"text-{color}")], className="small mb-3"),
                
                html.Div([
                    html.Small("Battery", className="text-muted"),
                    dbc.Progress(value=battery, color="info", className="mb-2", style={"height": "5px"}),
                    html.Small(f"Uptime: {uptime} hrs", className="text-muted float-end")
                ])
            ]),
            className=f"glass-panel border-0 border-top border-{color} border-3 h-100"
        ),
        width=12, sm=6, lg=4, xl=3, className="mb-4"
    )

layout = html.Div(
    [
        html.Div(
            [
                html.H1("Robot Fleet Status", className="display-5 fw-bold text-gradient mb-2"),
                html.P(
                    "Live monitoring of all factory floor robots and their tactile sensor health.",
                    className="lead text-secondary",
                ),
            ],
            className="mb-4 pb-3 border-bottom border-secondary border-opacity-25",
        ),
        
        dbc.Row([
            dbc.Col(dbc.Button([html.I(className="bi bi-arrow-clockwise me-2"), "Refresh Status"], color="primary", outline=True, className="mb-4"), width="auto"),
            dbc.Col(html.Div([
                html.Span(className="badge bg-success me-2", children="124 Online"),
                html.Span(className="badge bg-warning me-2", children="2 Calibrating"),
                html.Span(className="badge bg-danger", children="4 Offline")
            ], className="text-end pt-2"))
        ]),
        
        dbc.Row(
            [
                generate_robot_card("1042", "Active", "SMT Assembly - Alpha", 85, 12.4),
                generate_robot_card("1043", "Active", "SMT Assembly - Alpha", 78, 12.4),
                generate_robot_card("1088", "Calibrating", "Inspection - Beta", 92, 0.2),
                generate_robot_card("1091", "Active", "Packaging - Gamma", 45, 8.1),
                generate_robot_card("1105", "Offline", "Heavy Payload - Delta", 12, 0.0),
                generate_robot_card("1106", "Active", "Heavy Payload - Delta", 88, 5.5),
                generate_robot_card("1122", "Active", "SMT Assembly - Beta", 67, 9.2),
                generate_robot_card("1123", "Active", "SMT Assembly - Beta", 95, 2.1),
            ]
        )
    ],
    className="animate__animated animate__fadeIn",
)
