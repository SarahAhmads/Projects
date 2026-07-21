"""
Main Application Entry Point
Ford GoBike Interactive Dashboard
"""

import dash
from dash import html
from components.filters  import create_sidebar
from components.kpi_cards import create_kpi_cards
from components.charts   import create_charts

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Ford GoBike Dashboard"

app.layout = html.Div([
    # ── Fixed top filter bar ───────────────────────────────────────────────
    create_sidebar(),

    # ── Main content below the bar ─────────────────────────────────────────
    html.Div([
        html.Div(create_kpi_cards(), className="kpi-row"),
        *create_charts(),
    ], className="content")
])

import callbacks  # noqa: E402 – must be imported after app is defined

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8055, debug=True)