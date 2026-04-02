import json
import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ─── Load Data ────────────────────────────────────────────────────────────────
with open("data/hospital_data.json") as f:
    data = json.load(f)

hospital = data["hospital"]
departments = data["departments"]

# Build flat DataFrames
patients_rows = []
staff_rows = []
for dept in departments:
    for p in dept["patients"]:
        patients_rows.append({
            "Department": dept["name"],
            "Name": p["name"],
            "Age": p["age"],
            "Medical Record": p["medical_record"],
        })
    for s in dept["staff"]:
        staff_rows.append({
            "Department": dept["name"],
            "Name": s["name"],
            "Age": s["age"],
            "Position": s["position"],
        })

df_patients = pd.DataFrame(patients_rows)
df_staff = pd.DataFrame(staff_rows)

dept_names = [d["name"] for d in departments]
dept_patient_counts = [len(d["patients"]) for d in departments]
dept_staff_counts = [len(d["staff"]) for d in departments]

total_patients = len(df_patients)
total_staff = len(df_staff)
total_depts = len(departments)
avg_patient_age = round(df_patients["Age"].mean(), 1)

# ─── Colors ──────────────────────────────────────────────────────────────────
COLORS = {
    "bg": "#0f172a",
    "card": "#1e293b",
    "border": "#334155",
    "accent": "#38bdf8",
    "accent2": "#818cf8",
    "accent3": "#34d399",
    "accent4": "#f472b6",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
}

DEPT_COLORS = ["#38bdf8", "#818cf8", "#34d399", "#f472b6"]

# ─── Charts ──────────────────────────────────────────────────────────────────
def make_bar_chart():
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Patients", x=dept_names, y=dept_patient_counts,
        marker_color=COLORS["accent"], marker_line_color="rgba(0,0,0,0)", opacity=0.9,
        text=dept_patient_counts, textposition="outside", textfont=dict(color=COLORS["text"])
    ))
    fig.add_trace(go.Bar(
        name="Staff", x=dept_names, y=dept_staff_counts,
        marker_color=COLORS["accent2"], marker_line_color="rgba(0,0,0,0)", opacity=0.9,
        text=dept_staff_counts, textposition="outside", textfont=dict(color=COLORS["text"])
    ))
    fig.update_layout(
        barmode="group", plot_bgcolor=COLORS["card"], paper_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        xaxis=dict(gridcolor=COLORS["border"], title="Department"),
        yaxis=dict(gridcolor=COLORS["border"], title="Count"),
        legend=dict(bgcolor=COLORS["card"], bordercolor=COLORS["border"]),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig

def make_donut(title, values, labels, colors):
    fig = go.Figure(go.Pie(
        values=values, labels=labels, hole=0.62,
        marker=dict(colors=colors, line=dict(color=COLORS["bg"], width=3)),
        textinfo="percent", textfont=dict(size=13, color=COLORS["text"]),
        hovertemplate="<b>%{label}</b><br>%{value} people<extra></extra>"
    ))
    fig.update_layout(
        plot_bgcolor=COLORS["card"], paper_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        showlegend=True,
        legend=dict(bgcolor=COLORS["card"], bordercolor=COLORS["border"],
                    font=dict(size=11)),
        margin=dict(l=10, r=10, t=10, b=10),
        annotations=[dict(text=f"<b>{title}</b>", x=0.5, y=0.5,
                          font=dict(size=14, color=COLORS["text"]),
                          showarrow=False)]
    )
    return fig

def make_scatter():
    fig = go.Figure()
    for i, dept in enumerate(departments):
        ages_p = [p["age"] for p in dept["patients"]]
        ages_s = [s["age"] for s in dept["staff"]]
        names_p = [p["name"] for p in dept["patients"]]
        names_s = [s["name"] for s in dept["staff"]]
        fig.add_trace(go.Scatter(
            x=[dept["name"]] * len(ages_p), y=ages_p,
            mode="markers", name=f"{dept['name']} Patients",
            marker=dict(symbol="circle", size=14, color=DEPT_COLORS[i], opacity=0.8,
                        line=dict(color=COLORS["bg"], width=1)),
            text=names_p, hovertemplate="<b>%{text}</b><br>Age: %{y}<br>Patient<extra></extra>",
            legendgroup=dept["name"]
        ))
        fig.add_trace(go.Scatter(
            x=[dept["name"]] * len(ages_s), y=ages_s,
            mode="markers", name=f"{dept['name']} Staff",
            marker=dict(symbol="diamond", size=13, color=DEPT_COLORS[i], opacity=0.5,
                        line=dict(color=COLORS["bg"], width=1)),
            text=names_s, hovertemplate="<b>%{text}</b><br>Age: %{y}<br>Staff<extra></extra>",
            legendgroup=dept["name"]
        ))
    fig.update_layout(
        plot_bgcolor=COLORS["card"], paper_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        xaxis=dict(gridcolor=COLORS["border"], title="Department"),
        yaxis=dict(gridcolor=COLORS["border"], title="Age", range=[0, 70]),
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(bgcolor=COLORS["card"], bordercolor=COLORS["border"], font=dict(size=10))
    )
    return fig

# ─── App Layout ──────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="Cairo Hospital Dashboard")

def kpi_card(value, label, color):
    return html.Div([
        html.Div(str(value), style={
            "fontSize": "2.4rem", "fontWeight": "700",
            "color": color, "lineHeight": "1"
        }),
        html.Div(label, style={"fontSize": "0.8rem", "color": COLORS["muted"],
                                "marginTop": "6px", "textTransform": "uppercase",
                                "letterSpacing": "0.1em"})
    ], style={
        "background": COLORS["card"], "border": f"1px solid {COLORS['border']}",
        "borderTop": f"3px solid {color}",
        "borderRadius": "10px", "padding": "22px 24px",
        "flex": "1", "minWidth": "140px"
    })

def section_card(title, children):
    return html.Div([
        html.Div(title, style={
            "fontSize": "0.75rem", "fontWeight": "600", "color": COLORS["muted"],
            "textTransform": "uppercase", "letterSpacing": "0.1em", "marginBottom": "14px"
        }),
        *children
    ], style={
        "background": COLORS["card"], "border": f"1px solid {COLORS['border']}",
        "borderRadius": "10px", "padding": "20px 22px"
    })

TABLE_STYLE = dict(
    style_table={"overflowX": "auto"},
    style_cell={
        "backgroundColor": COLORS["card"], "color": COLORS["text"],
        "border": f"1px solid {COLORS['border']}",
        "padding": "10px 14px", "fontFamily": "Inter, sans-serif", "fontSize": "13px"
    },
    style_header={
        "backgroundColor": COLORS["bg"], "color": COLORS["accent"],
        "fontWeight": "700", "border": f"1px solid {COLORS['border']}",
        "textTransform": "uppercase", "fontSize": "11px", "letterSpacing": "0.08em"
    },
    style_data_conditional=[
        {"if": {"row_index": "odd"}, "backgroundColor": "#182032"}
    ],
    page_size=8,
)

app.layout = html.Div([
    # ── Header ──────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("🏥", style={"fontSize": "1.8rem", "marginRight": "12px"}),
            html.Div([
                html.Div(hospital["name"], style={
                    "fontSize": "1.5rem", "fontWeight": "700", "color": COLORS["text"]
                }),
                html.Div(hospital["location"], style={
                    "fontSize": "0.85rem", "color": COLORS["muted"]
                }),
            ])
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div("Hospital Management Dashboard", style={
            "fontSize": "0.8rem", "color": COLORS["accent"],
            "fontWeight": "600", "textTransform": "uppercase", "letterSpacing": "0.12em"
        })
    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "20px 32px", "borderBottom": f"1px solid {COLORS['border']}",
        "background": COLORS["card"]
    }),

    # ── Body ────────────────────────────────────────────────
    html.Div([

        # KPI Row
        html.Div([
            kpi_card(total_patients, "Total Patients", COLORS["accent"]),
            kpi_card(total_staff, "Total Staff", COLORS["accent2"]),
            kpi_card(total_depts, "Departments", COLORS["accent3"]),
            kpi_card(avg_patient_age, "Avg Patient Age", COLORS["accent4"]),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                  "marginBottom": "20px"}),

        # Bar + Donuts Row
        html.Div([
            html.Div([
                section_card("Patients & Staff per Department", [
                    dcc.Graph(figure=make_bar_chart(), style={"height": "280px"},
                              config={"displayModeBar": False})
                ])
            ], style={"flex": "2", "minWidth": "340px"}),

            html.Div([
                section_card("Patients Distribution", [
                    dcc.Graph(
                        figure=make_donut("Patients",
                                          dept_patient_counts, dept_names, DEPT_COLORS),
                        style={"height": "240px"}, config={"displayModeBar": False})
                ])
            ], style={"flex": "1", "minWidth": "220px"}),

            html.Div([
                section_card("Staff Distribution", [
                    dcc.Graph(
                        figure=make_donut("Staff",
                                          dept_staff_counts, dept_names, DEPT_COLORS),
                        style={"height": "240px"}, config={"displayModeBar": False})
                ])
            ], style={"flex": "1", "minWidth": "220px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                  "marginBottom": "20px"}),

        # Age Scatter
        html.Div([
            section_card("Age Distribution by Department (● Patients  ◆ Staff)", [
                dcc.Graph(figure=make_scatter(), style={"height": "280px"},
                          config={"displayModeBar": False})
            ])
        ], style={"marginBottom": "20px"}),

        # Tables Row
        html.Div([
            html.Div([
                section_card("", [
                    html.Div([
                        html.Div("Filter Department:", style={
                            "color": COLORS["muted"], "fontSize": "0.8rem",
                            "marginBottom": "8px"
                        }),
                        dcc.Dropdown(
                            id="dept-filter",
                            options=[{"label": "All Departments", "value": "ALL"}] +
                                    [{"label": d, "value": d} for d in dept_names],
                            value="ALL",
                            clearable=False,
                            style={"marginBottom": "16px"},
                        ),
                    ]),
                    html.Div("Patients", style={
                        "fontSize": "0.75rem", "fontWeight": "600",
                        "color": COLORS["accent"], "marginBottom": "8px",
                        "textTransform": "uppercase"
                    }),
                    dash_table.DataTable(
                        id="patients-table",
                        columns=[{"name": c, "id": c} for c in df_patients.columns],
                        data=df_patients.to_dict("records"),
                        **TABLE_STYLE
                    ),
                ])
            ], style={"flex": "1", "minWidth": "300px"}),

            html.Div([
                section_card("", [
                    html.Div(style={"height": "72px"}),  # spacer to align with filter
                    html.Div("Staff", style={
                        "fontSize": "0.75rem", "fontWeight": "600",
                        "color": COLORS["accent2"], "marginBottom": "8px",
                        "textTransform": "uppercase"
                    }),
                    dash_table.DataTable(
                        id="staff-table",
                        columns=[{"name": c, "id": c} for c in df_staff.columns],
                        data=df_staff.to_dict("records"),
                        **TABLE_STYLE
                    ),
                ])
            ], style={"flex": "1", "minWidth": "300px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

    ], style={"padding": "24px 32px", "maxWidth": "1400px", "margin": "0 auto"}),

], style={
    "fontFamily": "Inter, -apple-system, sans-serif",
    "background": COLORS["bg"],
    "minHeight": "100vh",
    "color": COLORS["text"]
})

# ─── Callbacks ────────────────────────────────────────────────────────────────
@app.callback(
    Output("patients-table", "data"),
    Output("staff-table", "data"),
    Input("dept-filter", "value")
)
def filter_tables(dept):
    if dept == "ALL":
        return df_patients.to_dict("records"), df_staff.to_dict("records")
    return (
        df_patients[df_patients["Department"] == dept].to_dict("records"),
        df_staff[df_staff["Department"] == dept].to_dict("records"),
    )

if __name__ == "__main__":
    app.run(debug=False, port=8050)
