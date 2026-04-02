from dash import html, dcc

def create_sidebar():
    """
    Horizontal Top Filter Bar with 5 filters:
    User Type, Gender, Age Group, Weekday, Hour of Day
    """
    return html.Div(
        [
            # Brand
            html.Div("🚲 Ford GoBike", className="topbar-brand", style={"flex": "0 0 auto"}),

            # Filter 1: User Type
            html.Div([
                html.Label("User Type"),
                dcc.Dropdown(
                    id='user-type-filter',
                    options=[
                        {'label': 'All',        'value': 'All'},
                        {'label': 'Subscriber', 'value': 'Subscriber'},
                        {'label': 'Customer',   'value': 'Customer'},
                    ],
                    value='All', clearable=False,
                    style={'width': '130px', 'fontSize': '0.95rem'}
                ),
            ], className="filter-group", style={"flex": "1"}),

            # Filter 2: Gender
            html.Div([
                html.Label("Gender"),
                dcc.Dropdown(
                    id='gender-filter',
                    options=[
                        {'label': 'All',    'value': 'All'},
                        {'label': 'Male',   'value': 'Male'},
                        {'label': 'Female', 'value': 'Female'},
                        {'label': 'Other',  'value': 'Other'},
                    ],
                    value='All', clearable=False,
                    style={'width': '120px', 'fontSize': '0.95rem'}
                ),
            ], className="filter-group", style={"flex": "1"}),

            # Filter 3: Age Group
            html.Div([
                html.Label("Age Group"),
                dcc.Dropdown(
                    id='age-group-filter',
                    options=[
                        {'label': 'All',   'value': 'All'},
                        {'label': '25-34', 'value': '25-34'},
                        {'label': '35-44', 'value': '35-44'},
                        {'label': '45-54', 'value': '45-54'},
                        {'label': '55-64', 'value': '55-64'},
                        {'label': '65-80', 'value': '65-80'},
                    ],
                    value='All', clearable=False,
                    style={'width': '120px', 'fontSize': '0.95rem'}
                ),
            ], className="filter-group", style={"flex": "1"}),

            # Filter 4: Hour of Day
            html.Div([
                html.Label("Hour of Day"),
                dcc.RangeSlider(
                    id='hour-slider',
                    min=0, max=23, step=1,
                    marks={i: f'{i}h' for i in range(0, 24, 6)},
                    value=[0, 23],
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
            ], className="filter-group-slider", style={"flex": "2"}),
        ],
        className="topbar"
    )