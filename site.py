# creating dash

from dash import Dash, dcc, html

app = Dash(__name__)

app.layout = html.Div(
    children=[
        html.H1(children="Wikipedia analytics"),
        html.P(
            children=(
                "Analyze the rate each word appears"
                " in the wikipedia article on democracy"
            ),
        ),
        dcc.Graph(
            figure={
                "data": [
                    {
                        "x": ['democracy', 'political', 'government', 'democratic', 'state', 'right', 'citizen', 'power', 'country', 'form', 'people', 'representative', 'system', 'rule', 'vote', 'republic', 'election', 'medium', 'may', 'also', 'direct', 'voting', 'economic', 'time', 'freedom', 'example', 'elected', 'social'],
                        "y": [292, 88, 77, 66, 60, 57, 54, 48, 43, 42, 41, 39, 38, 37, 36, 34, 34, 30, 30, 30, 29, 27, 26, 24, 24, 24, 24, 23],
                        "type": "bars",
                    },
                ],
                "layout": {"title": "Rate each word appears"},
            },
        ),
        dcc.Graph(
            figure={
                "data": [
                    {
                        "x": ['democracy', 'political', 'government', 'democratic', 'state', 'right', 'citizen', 'power', 'country', 'form', 'people', 'representative', 'system', 'rule', 'vote', 'republic', 'election', 'medium', 'may', 'also', 'direct', 'voting', 'economic', 'time', 'freedom', 'example', 'elected', 'social'],
                        "y": [292, 88, 77, 66, 60, 57, 54, 48, 43, 42, 41, 39, 38, 37, 36, 34, 34, 30, 30, 30, 29, 27, 26, 24, 24, 24, 24, 23],
                        "type": "bars",
                    },
                ],
                "layout": {"title": "Rate each word appears"},
            },
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=False)