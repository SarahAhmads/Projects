from dash import html

def create_kpi_cards():
    """4 KPI cards with icons."""
    cards = [
        ("🚴", "Total Trips",     "total-trips-kpi",    "–"),
        ("⏱️", "Avg Duration",    "avg-duration-kpi",   "–"),
        ("👥", "Subscribers",     "subscribers-kpi",    "–"),
        ("📍", "Active Stations", "active-stations-kpi","–"),
    ]
    return [
        html.Div([
            html.Div([
                html.Span(icon, style={"fontSize": "1.4rem"}),
                html.Div([
                    html.Div(label, className="kpi-title"),
                    html.Div(id=kpi_id, className="kpi-value", children=default),
                ], style={"marginLeft": "10px"})
            ], style={"display": "flex", "alignItems": "center"}),
        ], className="card-custom")
        for icon, label, kpi_id, default in cards
    ]