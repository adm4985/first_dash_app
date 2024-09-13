from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')

# Initialize the app
app = Dash()

# Layout of the app
app.layout = [
    html.H1(children='Title of Dash App', style={'textAlign': 'center'}),
    # Dropdown for selecting country
    dcc.Dropdown(
        id='dropdown-country',
        options=[{'label': country, 'value': country} for country in df.country.unique()],
        value='China'  # Default value
    ),
    # Dropdown for selecting year
    dcc.Dropdown(
        id='dropdown-year',
        options=[],
        value=None  # Default value (None means no initial year selected)
    ),
    dcc.Graph(id='graph-content')
]

@callback(
    Output('dropdown-year', 'options'),
    Input('dropdown-country', 'value')
)
def update_year_dropdown(selected_country):
    # Filter the dataframe based on the selected country
    filtered_df = df[df.country == selected_country]
    # Generate a list of years for the selected country
    years = [{'label': year, 'value': year} for year in filtered_df.year.unique()]
    return years

@callback(
    Output('graph-content', 'figure'),
    [Input('dropdown-country', 'value'),
     Input('dropdown-year', 'value')]
)
def update_graph(selected_country, selected_year):
    # Filter the dataframe based on the selected country and year
    filtered_df = df[(df.country == selected_country) & (df.year == selected_year)]
    # Generate the figure
    fig = px.line(filtered_df, x='year', y='pop', title=f'Population of {selected_country} in {selected_year}')
    return fig

if __name__ == '__main__':
    app.run(debug=True)