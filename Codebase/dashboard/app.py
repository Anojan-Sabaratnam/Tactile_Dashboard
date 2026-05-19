import os

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[
        dbc.themes.FLATLY, 
        dbc.icons.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Roboto+Mono:wght@400;500;700&display=swap"
    ],
    suppress_callback_exceptions=True,
    title="AIFI Tactile Dashboard",
)

sidebar = html.Div(
    [
        html.Div(
            [
                html.H2("AIFI", className="display-4 text-primary fw-bold mb-0"),
                html.P("Tactile Intelligence", className="lead text-muted fst-italic"),
            ],
            className="mb-4",
        ),
        html.Hr(className="border-primary opacity-50"),
        html.P(
            "Industry 5.0 Humanoid Robotics", className="text-secondary small text-uppercase fw-bold"
        ),
        dbc.Nav(
            [
                dbc.NavLink(
                    [html.I(className="bi bi-grid-fill me-2"), "Fleet Overview"],
                    href="/",
                    active="exact",
                    className="fw-semibold rounded-3 mb-2 shadow-sm transition-all",
                ),
                dbc.NavLink(
                    [html.I(className="bi bi-robot me-2"), "Robot Fleet Status"],
                    href="/fleet-status",
                    active="exact",
                    className="fw-semibold rounded-3 mb-2 shadow-sm transition-all",
                ),
                dbc.NavLink(
                    [html.I(className="bi bi-activity me-2"), "Edge AI Telemetry"],
                    href="/live-demo",
                    active="exact",
                    className="fw-semibold rounded-3 mb-2 shadow-sm transition-all",
                ),
                dbc.NavLink(
                    [html.I(className="bi bi-graph-up me-2"), "Historical Analytics"],
                    href="/historical-analytics",
                    active="exact",
                    className="fw-semibold rounded-3 mb-2 shadow-sm transition-all",
                ),
                dbc.NavLink(
                    [html.I(className="bi bi-wrench-adjustable me-2"), "AI Diagnostics"],
                    href="/model-comparison",
                    active="exact",
                    className="fw-semibold rounded-3 mb-2 shadow-sm transition-all",
                ),
                dbc.NavLink(
                    [html.I(className="bi bi-cash-coin me-2"), "ROI & System Config"],
                    href="/business-impact",
                    active="exact",
                    className="fw-semibold rounded-3 mb-2 shadow-sm transition-all",
                ),
            ],
            vertical=True,
            pills=True,
            className="mt-4 gap-2",
        ),
    ],
    className="sidebar bg-light px-4 py-5 shadow-sm border-end border-secondary border-opacity-10 h-100",
    style={"position": "fixed", "top": 0, "left": 0, "width": "280px", "zIndex": 1000},
)

content = html.Div(
    dash.page_container,
    className="content-container p-5",
    style={"marginLeft": "280px", "minHeight": "100vh"},
)

app.layout = html.Div([sidebar, content], className="bg-light")

if __name__ == "__main__":
    app.run(debug=os.getenv("DASH_DEBUG", "false").lower() == "true", port=8050)
