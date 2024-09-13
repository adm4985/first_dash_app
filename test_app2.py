from dash import Dash, html, dcc, callback, Output, Input, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

#process data and create dataframes
df = pd.read_csv('norCalResults.csv')
df = df.sort_values(by='TEAM')
df_summary = df.groupby(['LEVEL_REGION','GAME_TYPE','TEAM']).agg({'WIN':'sum','LOSS':'sum','DRAW':'sum','TEAM_SCORE':'sum','OPPONENT_SCORE':'sum','GOAL_DIFF':'sum'}).reset_index(drop=False)
df_summary['MP'] = df_summary['WIN']+df_summary['LOSS']+df_summary['DRAW']
df_summary['POINTS'] = df_summary['WIN'] * 3 + df_summary['DRAW']
df_summary = df_summary[['LEVEL_REGION','GAME_TYPE','TEAM','MP','WIN','LOSS','DRAW','TEAM_SCORE','OPPONENT_SCORE','GOAL_DIFF','POINTS']]
df_summary.sort_values(by='WIN', ascending=False)
team_pivot = df[ (df['LEVEL_REGION']=='Silver-Region 2') & (df['GAME_TYPE']=='FALL_LEAGUE')].groupby(['TEAM', 'OPPONENT'])['GAME_SCORE'].apply(lambda x: ', '.join(x.dropna()).strip(', ')).unstack(fill_value='').reset_index(drop=False)

 

# Initialize the app
#app = Dash(external_stylesheets=[dbc.themes.DARKLY])
app = Dash()
app.layout = [
    html.H1(children='Results', style={'textAlign': 'center'}),
    # Dropdown for selecting country
    dcc.Dropdown(
        id='dropdown-team',
        options=[{'label': TEAM, 'value': TEAM} for TEAM in df.TEAM.unique() if TEAM ==TEAM],
        value='Castro Valley Soccer Club Castro Valley SC CVSC United Green 2015 G' , # Default value
        style={'width': '500px'}
    ),
    html.Div(style={'height': '50px'}),
   dcc.Dropdown(
        id='dropdown-game_type',
        options=[{'label': GAME_TYPE, 'value': GAME_TYPE} for GAME_TYPE in df.GAME_TYPE.unique() if GAME_TYPE ==GAME_TYPE],
        value=None , # Default value
        style={'width': '500px'}
    ),
   html.Div([
        html.Div(

            dash_table.DataTable(
                id='table-summary',
                data=df_summary.to_dict('records'),
                columns=[{"name": i, "id": i} for i in df_summary.columns],
                fill_width=False,
                style_header={
                        'backgroundColor': 'gray',
                        'whiteSpace': 'normal',  # Allows text to wrap
                        'height': 'auto',  # Adjust height to fit content
                        'textAlign': 'center',  # Center-align text
                        'overflow': 'hidden',  # Prevent overflow
                        'textOverflow': 'ellipsis'  # Add ellipsis for overflowed text
                    },
                style_data={
                'whiteSpace': 'normal',
                'height': 'auto',
                },
                style_cell={'textAlign': 'center'},
                style_cell_conditional=[
                    {
                        'if': {'column_id': 'TEAM'},
                        'textAlign': 'left'
                    }],
                style_table={
                    'width': '80%',  # Adjust the width as needed
                    'maxWidth': '1000px',  # Set a maximum width for the table
                    'overflowX': 'auto'  # Enable horizontal scrolling if the content overflows
                }

            ) ,
            style={'display': 'inline-block'}
    	),
      
        
        html.Div(
            dash_table.DataTable(
                id='table-pivot',
                style_header={
                    'backgroundColor': 'gray',
                    'whiteSpace': 'normal',  # Allows text to wrap
                    'height': 'auto',  # Adjust height to fit content
                    'textAlign': 'center',  # Center-align text
                    'overflow': 'hidden',  # Prevent overflow
                    'textOverflow': 'ellipsis'  # Add ellipsis for overflowed text
                },
                style_table={
                'width': '80%',  # Adjust the width as needed
                'maxWidth': '1500px',  # Set a maximum width for the table
                'overflowX': 'auto'  # Enable horizontal scrolling if the content overflows
            }
            ),  style={'display': 'inline-block'}
        ) 
        ]),

    dcc.Graph(id='graph-wins'),
    #dcc.Graph(id='graph-goals')

    ]

@callback(
    Output('graph-wins','figure'),
    Output('table-summary','data'),
    Output('table-pivot','data'),
    Input('dropdown-team', 'value')
	)

def update_team_dropdown(selected_team):
   # Filter the dataframe based on the selected team
    selected_region_level = list(df['LEVEL_REGION'][df['TEAM'] == selected_team].drop_duplicates())[0]

    wins_df = df_summary[(df_summary['LEVEL_REGION'] == selected_region_level ) & (df_summary['GAME_TYPE']=='FALL_LEAGUE')].sort_values(by='WIN', ascending=False)
    table_data = wins_df.to_dict('records')

    # Create the pivot table
    pivot_data = df[(df['LEVEL_REGION'] == selected_region_level) &  (df['GAME_TYPE']=='FALL_LEAGUE') ].groupby(['TEAM', 'OPPONENT'])['GAME_SCORE'] \
        .apply(lambda x: ', '.join(x.dropna()).strip(', ')) \
        .unstack(fill_value='') \
        .reset_index(drop=False)

    team_pivot_data = pivot_data.to_dict('records')
    
    # Generate the figure
    wins_fig = px.bar(wins_df, x=['TEAM_SCORE','OPPONENT_SCORE'], y='TEAM', title=f'Goals Summary', labels={'TEAM_SCORE':'Goals Scored','OPPONENT_SCORE':'Goals Allowed'}, orientation='h', width=1200)
    
    return wins_fig, table_data , team_pivot_data

#grouped = t.groupby(['TEAM', 'OPPONENT'])['GAME_SCORE'].apply(lambda x: ', '.join(x.dropna()).strip(', ')).unstack(fill_value='')
if __name__ == '__main__':
    app.run(debug=True)