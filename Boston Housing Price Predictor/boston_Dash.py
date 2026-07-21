import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import pickle

# Load trained model
with open("boston_model.pkl", "rb") as f:
    model = pickle.load(f)

app = dash.Dash(__name__)

app.layout = html.Div(
    style={
        "backgroundColor": "#1e1e2f",
        "height": "100vh",
        "display": "flex",
        "justifyContent": "center",
        "alignItems": "center",
        "fontFamily": "Segoe UI"
    },

    children=[
        html.Div(

            style={
                "backgroundColor": "#2b2b3c",
                "padding": "40px",
                "borderRadius": "10px",
                "width": "400px",
                "boxShadow": "0px 5px 20px rgba(0,0,0,0.3)"
            },

            children=[

                html.H2(
                    "Boston Housing Predictor",
                    style={
                        "color": "white",
                        "textAlign": "center",
                        "marginBottom": "30px"
                    }
                ),

                html.Label("Average Rooms (RM)", style={"color": "#ccc"}),
                dcc.Input(
                    id="rm",
                    type="number",
                    placeholder="Enter RM",
                    style={"width": "100%", "marginBottom": "15px"}
                ),

                html.Label("Lower Status % (LSTAT)", style={"color": "#ccc"}),
                dcc.Input(
                    id="lstat",
                    type="number",
                    placeholder="Enter LSTAT",
                    style={"width": "100%", "marginBottom": "15px"}
                ),

                html.Label("Pupil Teacher Ratio (PTRATIO)", style={"color": "#ccc"}),
                dcc.Input(
                    id="ptratio",
                    type="number",
                    placeholder="Enter PTRATIO",
                    style={"width": "100%", "marginBottom": "20px"}
                ),

                html.Button(
                    "Predict Price",
                    id="predict_btn",
                    n_clicks=0,
                    style={
                        "width": "100%",
                        "backgroundColor": "#4a90e2",
                        "color": "white",
                        "border": "none",
                        "padding": "10px",
                        "fontSize": "16px",
                        "borderRadius": "5px",
                        "cursor": "pointer"
                    }
                ),

                html.Div(
                    id="prediction_output",
                    style={
                        "marginTop": "20px",
                        "color": "#2ecc71",
                        "fontSize": "18px",
                        "textAlign": "center"
                    }
                )

            ]
        )
    ]
)

@app.callback(
    Output("prediction_output", "children"),
    Input("predict_btn", "n_clicks"),
    State("rm", "value"),
    State("lstat", "value"),
    State("ptratio", "value")
)

def predict_price(n_clicks, rm, lstat, ptratio):

    if n_clicks == 0:
        return ""

    if rm is None or lstat is None or ptratio is None:
        return "Please fill all fields"

    features = np.array([[rm, lstat, ptratio]])

    prediction = model.predict(features)

    return f"Predicted Price: ${prediction[0]*1000:,.2f}"


if __name__ == "__main__":
    app.run(debug=True)