from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

df = pd.read_csv('norCalResults.csv')

# Initialize the app
app = Dash()

# Layout of the app
app.layout = [
    html.H1(children='Title of Dash App', style={'textAlign': 'center'}),
    # Dropdown for selecting country
    dcc.Dropdown(
        id='dropdown-level',
        options=[{'label': LEVEL, 'value': LEVEL} for LEVEL in df.LEVEL.unique()],
        value=None , # Default value
        style={'width': '250px'}
    ),
        dcc.Dropdown(
        id='dropdown-team',
        options=[{'label': TEAM, 'value': TEAM} for TEAM in df.TEAM.unique() if TEAM ==TEAM],
        value=None , # Default value
        style={'width': '500px'}
    ),

    dcc.Graph(id='graph-wins'),
    dcc.Graph(id='graph-goals')
    ]

@callback(
    [Output('graph-wins', 'figure'),
    Output('graph-goals', 'figure')],
    [Input('dropdown-level', 'value'),
    Input('dropdown-team', 'value')]
)

def update_graph(selected_level,selected_team):
    # Filter the dataframe based on the selected country and year
    wins_df = df[(df.LEVEL == selected_level)]
    wins_df = wins_df.groupby('TEAM')['WIN'].sum().reset_index(drop=False)
    wins_df.columns = ['TEAM','WINS']
    # Generate the figure
    wins_fig = px.bar(wins_df, x='WINS', y='TEAM', title=f'Wins by {selected_level}',orientation = 'h', width=1200)
    
    #Goals
    team_level = df[(df.TEAM == selected_team)]
    level_team = team_level['LEVEL-REGION'].iloc[0]
    print(level_team)
    wins_df = df[(df['LEVEL-REGION'] == level_team)]
    wins_df = wins_df.groupby('TEAM')['WIN','LOSS'].sum().reset_index(drop=False)
    wins_df.columns = ['TEAM','WINS']
    # Generate the figure
    wins_fig = px.bar(wins_df, x='WINS', y='TEAM', title=f'Wins by {level_team}',orientation = 'h', width=1200)


    goals_df = df[(df['LEVEL-REGION'] == level_team)]
    goals_df = goals_df.groupby('TEAM')['TEAM_SCORE'].sum().reset_index(drop=False)
    goals_df.columns = ['TEAM','GOALS']
    goals_fig = px.bar(goals_df, x='GOALS', y='TEAM', title=f'GOALS by {level_team}',orientation = 'h', width=1200)
    return wins_fig, goals_fig

if __name__ == '__main__':
    app.run(debug=True)