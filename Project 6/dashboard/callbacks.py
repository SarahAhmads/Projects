import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, callback

# ── Elegant Palette ───────────────────────────────────────────────────────────
P = {
    'indigo':   '#6366f1',
    'cyan':     '#06b6d4',
    'pink':     '#ec4899',
    'amber':    '#f59e0b',
    'emerald':  '#10b981',
    'slate':    '#475569',
    'light':    '#f8fafc',
    'grid':     '#f1f5f9',
    'text':     '#1e293b',
}

# ── Load data ─────────────────────────────────────────────────────────────────
_BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_PATH = os.path.join(_BASE, 'data', 'processed', 'cleaned_fordgobike_data.csv')

df = pd.read_csv(_DATA_PATH)
df['start_time']  = pd.to_datetime(df['start_time'], errors='coerce')
df['start_hour']  = df['start_time'].dt.hour
df['day_of_week'] = df['start_time'].dt.day_name()

WDAY = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
WDAY_SHORT = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']


@callback(
    [
        Output('total-trips-kpi',       'children'),
        Output('avg-duration-kpi',      'children'),
        Output('subscribers-kpi',       'children'),
        Output('active-stations-kpi',   'children'),
        Output('time-analysis-chart',   'figure'),
        Output('user-behavior-chart',   'figure'),
        Output('station-analysis-chart','figure'),
    ],
    [
        Input('user-type-filter', 'value'),
        Input('gender-filter',    'value'),
        Input('age-group-filter', 'value'),
        Input('hour-slider',      'value'),
    ]
)
def update_dashboard(sel_user, sel_gender, sel_age, sel_hour):

    fdf = df.copy()
    if sel_user   != 'All': fdf = fdf[fdf['user_type']     == sel_user]
    if sel_gender != 'All': fdf = fdf[fdf['member_gender'] == sel_gender]
    if sel_age    != 'All': fdf = fdf[fdf['age_group'].astype(str) == sel_age]
    if sel_hour:
        fdf = fdf[(fdf['start_hour'] >= sel_hour[0]) & (fdf['start_hour'] <= sel_hour[1])]

    # ── Shared layout: clean, minimal, no clutter ─────────────────────────────
    LY = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color=P['slate'], size=10),
        margin=dict(l=35, r=15, t=30, b=30),
    )
    AXIS = dict(
        showgrid=True,
        gridcolor=P['grid'],
        gridwidth=1,
        zeroline=False,
        showline=False,
    )

    # ── Empty state ───────────────────────────────────────────────────────────
    if fdf.empty:
        empty = go.Figure()
        empty.update_layout(
            title=dict(text="No data matches filters", font=dict(size=11, color=P['slate'])),
            **LY
        )
        return "0", "–", "0%", "0", empty, empty, empty

    # ── KPIs ──────────────────────────────────────────────────────────────────
    total    = f"{len(fdf):,}"
    avg_dur  = f"{fdf['duration_mins'].mean():.1f} min" if 'duration_mins' in fdf.columns else "–"
    subs     = f"{(fdf['user_type']=='Subscriber').mean()*100:.1f}%"
    stations = f"{fdf['start_station_name'].nunique():,}" if 'start_station_name' in fdf.columns else "–"

    # ══════════════════════════════════════════════════════════════════════════
    # CHART 1 – Time Analysis (Weekday bar + Hour area, side by side)
    # ══════════════════════════════════════════════════════════════════════════
    fig_time = make_subplots(
        rows=1, cols=2,
        column_widths=[0.48, 0.52],
        horizontal_spacing=0.1,
        subplot_titles=("By Weekday", "By Hour")
    )

    wd = fdf.groupby('day_of_week').size().reindex(WDAY, fill_value=0).reset_index(name='trips')
    bar_colors = [P['pink'] if d in ['Saturday','Sunday'] else P['indigo'] for d in wd['day_of_week']]

    fig_time.add_trace(go.Bar(
        x=WDAY_SHORT, y=wd['trips'],
        marker=dict(color=bar_colors, cornerradius=4),
        hovertemplate='%{x}: <b>%{y:,}</b><extra></extra>',
        showlegend=False,
    ), row=1, col=1)

    hr = fdf.groupby('start_hour').size().reset_index(name='trips')
    fig_time.add_trace(go.Scatter(
        x=hr['start_hour'], y=hr['trips'],
        mode='lines',
        line=dict(color=P['cyan'], width=2.5, shape='spline'),
        fill='tozeroy',
        fillcolor='rgba(6,182,212,0.08)',
        hovertemplate='%{x}:00 → <b>%{y:,}</b><extra></extra>',
        showlegend=False,
    ), row=1, col=2)

    fig_time.update_layout(**LY, showlegend=False)
    fig_time.update_xaxes(**AXIS)
    fig_time.update_yaxes(**AXIS)
    fig_time.update_annotations(font=dict(size=10, color=P['slate'], family='Inter'))

    # ══════════════════════════════════════════════════════════════════════════
    # CHART 2 – User Analysis (Donut + Gender bar, side by side)
    # ══════════════════════════════════════════════════════════════════════════
    fig_user = make_subplots(
        rows=1, cols=2,
        column_widths=[0.45, 0.55],
        horizontal_spacing=0.08,
        subplot_titles=("User Type", "Gender"),
        specs=[[{"type": "pie"}, {"type": "bar"}]]
    )

    uc = fdf['user_type'].value_counts()
    fig_user.add_trace(go.Pie(
        labels=uc.index, values=uc.values,
        hole=0.55,
        marker=dict(colors=[P['indigo'], P['cyan']], line=dict(width=0)),
        textinfo='percent+label',
        textfont=dict(size=9, color='#ffffff'),
        insidetextorientation='horizontal',
        hovertemplate='<b>%{label}</b><br>%{value:,} (%{percent})<extra></extra>',
    ), row=1, col=1)

    gc = fdf['member_gender'].value_counts().reset_index()
    gc.columns = ['gender','count']
    gender_colors = {'Male': P['indigo'], 'Female': P['pink'], 'Other': P['amber']}
    fig_user.add_trace(go.Bar(
        x=gc['gender'], y=gc['count'],
        marker=dict(
            color=[gender_colors.get(g, P['emerald']) for g in gc['gender']],
            cornerradius=4
        ),
        hovertemplate='<b>%{x}</b>: %{y:,}<extra></extra>',
        showlegend=False,
    ), row=1, col=2)

    fig_user.update_layout(**LY, showlegend=False)
    fig_user.update_xaxes(**AXIS)
    fig_user.update_yaxes(**AXIS)
    fig_user.update_annotations(font=dict(size=10, color=P['slate'], family='Inter'))

    # ══════════════════════════════════════════════════════════════════════════
    # CHART 3 – Station Analysis (Top 8 horizontal bar)
    # ══════════════════════════════════════════════════════════════════════════
    ts = fdf['start_station_name'].value_counts().head(8).reset_index()
    ts.columns = ['station','trips']
    ts['short'] = ts['station'].apply(lambda s: (s[:30] + '…') if len(s) > 30 else s)

    fig_station = go.Figure(go.Bar(
        x=ts['trips'], y=ts['short'],
        orientation='h',
        marker=dict(
            color=ts['trips'],
            colorscale=[[0, '#e0e7ff'], [1, P['indigo']]],
            cornerradius=4,
            line=dict(width=0),
        ),
        hovertemplate='<b>%{y}</b><br>Trips: %{x:,}<extra></extra>',
    ))
    fig_station.update_layout(
        yaxis=dict(categoryorder='total ascending', tickfont=dict(size=9)),
        **LY
    )
    fig_station.update_xaxes(**AXIS)
    fig_station.update_yaxes(showgrid=False, zeroline=False, showline=False)

    return total, avg_dur, subs, stations, fig_time, fig_user, fig_station