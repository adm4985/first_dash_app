from dash import Dash, html, dcc, callback, Output, Input, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import networkx as nx
import io
import base64
import dash_bootstrap_components as dbc

#process data and create dataframes
df = pd.read_csv('norCalResults.csv')
df = df.sort_values(by='TEAM')
df_summary_fl = df[df.GAME_TYPE=='FALL_LEAGUE'].groupby(['LEVEL_REGION', 'GOLD_CUP_LEVEL_REGION', 'GAME_TYPE','TEAM']).agg({'WIN':'sum','LOSS':'sum','DRAW':'sum','TEAM_SCORE':'sum','OPPONENT_SCORE':'sum','GOAL_DIFF':'sum'}).reset_index(drop=False)
df_summary_sc = df[df.GAME_TYPE=='STATE_CUP'].groupby(['GOLD_CUP_LEVEL_REGION','LEVEL_REGION','GAME_TYPE','TEAM']).agg({'WIN':'sum','LOSS':'sum','DRAW':'sum','TEAM_SCORE':'sum','OPPONENT_SCORE':'sum','GOAL_DIFF':'sum'}).reset_index(drop=False)
df_summary_sc = df_summary_sc.rename(columns={'GOLD_CUP_LEVEL_REGION': 'LEVEL_REGION','LEVEL_REGION':'GOLD_CUP_LEVEL_REGION'})
df_summary = pd.concat([df_summary_fl, df_summary_sc], ignore_index=True)
df_summary['MP'] = df_summary['WIN']+df_summary['LOSS']+df_summary['DRAW']
df_summary['POINTS'] = df_summary['WIN'] * 3 + df_summary['DRAW']
df_summary = df_summary[['LEVEL_REGION','GOLD_CUP_LEVEL_REGION','GAME_TYPE','TEAM','MP','WIN','LOSS','DRAW','TEAM_SCORE','OPPONENT_SCORE','GOAL_DIFF','POINTS']]
df_summary.sort_values(by='WIN', ascending=False)
df_summary.to_csv('df_summary.csv')

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
   dcc.Dropdown(
        id='dropdown-game_type',
        options=[{'label': GAME_TYPE, 'value': GAME_TYPE} for GAME_TYPE in df.GAME_TYPE.unique() if GAME_TYPE ==GAME_TYPE],
        value='FALL_LEAGUE',
        style={'width': '500px'}
    ),
    dcc.Dropdown(
        id='dropdown-LEVEL_REGION',
        options=[{'label': LEVEL_REGION, 'value': LEVEL_REGION} for LEVEL_REGION in df_summary[df_summary.GAME_TYPE=='FALL_LEAGUE'].LEVEL_REGION.unique()],
        value=None , # Default value
        style={'width': '500px'}
    ),
    dcc.Dropdown(
	    id='dropdown-STATE_CUP',
	    options=[{'label': LEVEL_REGION, 'value': LEVEL_REGION} for LEVEL_REGION in df_summary[df_summary.GAME_TYPE=='STATE_CUP'].LEVEL_REGION.unique()],
	    value=None , # Default value
	    style={'width': '500px'}
    ),
   html.Div([
        html.Div(

            dash_table.DataTable(
                id='table-summary',
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

    html.Img(id='bar-graph-matplotlib')

    ]


#Call back to filter on team dropdown
@callback(
    Output('table-summary','data',allow_duplicate=True),
    Output('table-pivot','data',allow_duplicate=True),
    Input('dropdown-team', 'value'),
    Input('dropdown-game_type','value'),
    prevent_initial_call='initial_duplicate'
    )

def update_team_dropdown(selected_team,selected_game_type):
   # Filter the dataframe based on the selected team
    selected_region_level = list(df['LEVEL_REGION'][df['TEAM'] == selected_team].drop_duplicates())[0]
    selected_state_cup_level = list(df['GOLD_CUP_LEVEL_REGION'][df['TEAM'] == selected_team].drop_duplicates())[0]

    #map values based on game type
    mapped_columns_df = {
    'FALL_LEAGUE':selected_region_level,
    'STATE_CUP': selected_state_cup_level
    }

    #map columns based on game type
    mapped_columns_df_summary = {
    'FALL_LEAGUE':'LEVEL_REGION',
    'STATE_CUP': 'GOLD_CUP_LEVEL_REGION'}

    wins_df = df_summary[(df_summary['LEVEL_REGION'] == mapped_columns_df[selected_game_type] ) & (df_summary['GAME_TYPE']==selected_game_type)].sort_values(by='WIN', ascending=False)

    pivot_data = df[
    (df[mapped_columns_df_summary[selected_game_type]] == mapped_columns_df[selected_game_type] ) &
    (df['GAME_TYPE'] == selected_game_type)
    ].groupby(['TEAM', 'OPPONENT'])['GAME_SCORE'] \
    .apply(lambda x: ', '.join(x.dropna()).strip(', ')) \
    .unstack(fill_value='') \
    .reset_index(drop=False)

    wins_df = wins_df.drop(columns=['GAME_TYPE'])
    table_data = wins_df.to_dict('records')
    team_pivot_data = pivot_data.to_dict('records')
    return  table_data , team_pivot_data


if __name__ == '__main__':
    app.run(debug=True)