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
schedule_df = pd.read_csv('full_schedule.csv')
schedule_columns = ['DATETIME','TEAM_ONE','SCORE','TEAM_TWO','LOCATION','GAME_TYPE']
schedule_df = schedule_df[schedule_columns]
schedule_df.sort_values(by='DATETIME', ascending=True, inplace=True)


df = pd.read_csv('norCalResults.csv')
df = df.sort_values(by='TEAM')
df_summary_gt = df.groupby(['LEVEL_REGION', 'GOLD_CUP_LEVEL_REGION', 'GAME_TYPE','TEAM']).agg({'WIN':'sum','LOSS':'sum','DRAW':'sum','TEAM_SCORE':'sum','OPPONENT_SCORE':'sum','GOAL_DIFF':'sum'}).reset_index(drop=False)
df_summary_all = df.groupby(['LEVEL_REGION', 'GOLD_CUP_LEVEL_REGION', 'TEAM']).agg({'WIN':'sum','LOSS':'sum','DRAW':'sum','TEAM_SCORE':'sum','OPPONENT_SCORE':'sum','GOAL_DIFF':'sum'}).reset_index(drop=False)
df_summary = pd.concat([df_summary_gt, df_summary_all], ignore_index=True).copy()
df_summary['MP'] = df_summary['WIN']+df_summary['LOSS']+df_summary['DRAW']
df_summary['POINTS'] = df_summary['WIN'] * 3 + df_summary['DRAW']
df_summary = df_summary[['LEVEL_REGION','GOLD_CUP_LEVEL_REGION','GAME_TYPE','TEAM','MP','WIN','LOSS','DRAW','TEAM_SCORE','OPPONENT_SCORE','GOAL_DIFF','POINTS']]
df_summary.sort_values(by=['WIN','POINTS','GOAL_DIFF','LOSS'], ascending=[False,False,True,True])
df_summary['GAME_TYPE'] = df_summary['GAME_TYPE'].fillna('ALL_GAMES')
df_summary.to_csv('df_summary.csv')



#list for game types
gt = [GAME_TYPE for GAME_TYPE in df_summary.GAME_TYPE.unique() if GAME_TYPE ==GAME_TYPE]
gt.sort()

app = Dash(external_stylesheets=[dbc.themes.DARKLY])


app.layout = html.Div([
    html.H1(children='Results', style={'textAlign': 'center'}),

    

    # Container for dropdowns with a 2x2 grid layout
    html.Div([
        html.Div([
            html.H3('Select Team'),
            dcc.Dropdown(
                id='dropdown-team',
                options=[{'label': TEAM, 'value': TEAM} for TEAM in df.TEAM.unique() if TEAM == TEAM],
                value='Castro Valley Soccer Club Castro Valley SC CVSC United Green 2015 G',  # Default value
                style={
                    'width': '100%',
                    'border': '1px solid #ccc',  # Light gray border
                    'borderRadius': '5px',  # Rounded corners
                    'backgroundColor': 'white',  # Dark background color
                    'color': 'black',  # Text color for dropdown
                    'fontSize': '16px'  # Font size
                }
            )
        ], style={'padding': '10px', 'flex': '1', 'boxSizing': 'border-box'}),

        html.Div([
            html.H3('Select Game Type'),
            dcc.Dropdown(
                id='dropdown-game_type',
                options=[{'label': GAME_TYPE, 'value': GAME_TYPE} for GAME_TYPE in gt],
                value='FALL_LEAGUE',
                style={
                    'width': '100%',
                    'border': '1px solid #ccc',  # Light gray border
                    'borderRadius': '5px',  # Rounded corners
                    'backgroundColor': 'white',  # Dark background color
                    'color': 'black',  # Text color for dropdown
                    'fontSize': '16px'  # Font size
                }
            )
        ], style={'padding': '10px', 'flex': '1', 'boxSizing': 'border-box'}),

        html.Div([
            html.H3('Select Level Region'),
            dcc.Dropdown(
                id='dropdown-LEVEL_REGION',
                options=[{'label': LEVEL_REGION, 'value': LEVEL_REGION} for LEVEL_REGION in df_summary[df_summary.GAME_TYPE == 'FALL_LEAGUE'].LEVEL_REGION.unique()],
                value=None,  # Default value
                style={
                    'width': '100%',
                    'border': '1px solid #ccc',  # Light gray border
                    'borderRadius': '5px',  # Rounded corners
                    'backgroundColor': 'white',  # Dark background color
                    'color': 'black',  # Text color for dropdown
                    'fontSize': '16px'  # Font size
                }
            )
        ], style={'padding': '10px', 'flex': '1', 'boxSizing': 'border-box'}),

        html.Div([
            html.H3('Select State Cup'),
            dcc.Dropdown(
                id='dropdown-STATE_CUP',
                options=[{'label': GOLD_CUP_LEVEL_REGION, 'value': GOLD_CUP_LEVEL_REGION} for GOLD_CUP_LEVEL_REGION in df_summary[df_summary.GAME_TYPE == 'STATE_CUP'].GOLD_CUP_LEVEL_REGION.unique()],
                value=None,  # Default value
                style={
                    'width': '100%',
                    'border': '1px solid #ccc',  # Light gray border
                    'borderRadius': '5px',  # Rounded corners
                    'backgroundColor': 'white',  # Dark background color
                    'color': 'black',  # Text color for dropdown
                    'fontSize': '16px'  # Font size
                }
            )
        ], style={'padding': '10px', 'flex': '1', 'boxSizing': 'border-box'}),
    ], style={
        'display': 'flex',
        'flexWrap': 'wrap',
        'justifyContent': 'space-between',
        'gap': '10px'
    }),

    # Horizontal line between dropdowns and charts
    html.Hr(style={
        'border': '1px solid #ccc',  # Light gray border for the line
        'margin': '20px 0'  # Margin to create space above and below the line
    }),

     # Container for charts and tables in a 2x2 grid layout
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
                    'backgroundColor': 'darkgray',
                    'whiteSpace': 'normal',
                    'height': 'auto',
                },
                style_cell={'textAlign': 'center'},
                style_cell_conditional=[
                    {
                        'if': {'column_id': 'TEAM'},
                        'textAlign': 'left'
                    }

                ],


            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(105, 105, 105)',
                    
                }
                ],

                style_table={
                    'width': '100%',  # Adjust the width as needed
                    'maxWidth': '1500px',  # Set a maximum width for the table
                    'overflowX': 'auto'  # Enable horizontal scrolling if the content overflows
                }
            ),
            style={'padding': '10px'}
        ),

           html.Div(
            html.Img(id='bar-graph-matplotlib'),
            #style={'padding': '10px'}
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
                },
                style_data={
                    'backgroundColor': 'darkgray',
                    'whiteSpace': 'normal',
                    'height': 'auto'
                    }
            ),
            style={'padding': '10px'}
        ),

        html.Div(
            dash_table.DataTable(
            id='schedule-table',
            columns=[{"name": i, "id": i} for i in schedule_df.columns],
            data=schedule_df.to_dict('records'),
            style_table={'overflowX': 'auto','maxWidth': '1500px','height': '300px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'left'},
            style_data={
                    'backgroundColor': 'darkgray',
                    'whiteSpace': 'normal',
                    'height': 'auto'
                    },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(105, 105, 105)',
                }
                ],
            #page_size=10
 
            )
        ),
    ]
    #, style={
    #   'display': 'grid',
    #    'gridTemplateColumns': '1fr 1fr',
    #    'gap': '10px'
    #}
    )

 
])

# Helper functions
def winner_loser(team1, team2, score1, score2):
    return (team1, team2) if score1 > score2 else (team2, team1)

def plot_win_loss_graph(edges):
    plt.clf()
    plt.figure(figsize=(10, 8))
    plt.title('Win-Loss')
    G = nx.DiGraph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, k=5, iterations=100)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=750, edge_color='red', font_size=8, arrows=True)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("ascii")


#Call back to filter on team dropdown
@callback(
    Output('table-summary','data',allow_duplicate=True),
    Output('table-pivot','data',allow_duplicate=True),
    Output(component_id='bar-graph-matplotlib', component_property='src',allow_duplicate=True),
    Input('dropdown-team', 'value'),
    Input('dropdown-game_type','value'),
    prevent_initial_call='initial_duplicate'
	)

def update_team_dropdown(selected_team,selected_game_type):
    # Process the data for graph plotting
    df_union_full_score_set = df[(df.TEAM_SCORE == df.TEAM_SCORE) & (df.TEAM_SCORE != df.OPPONENT_SCORE)].copy()
   # Filter the dataframe based on the selected team
    selected_region_level = list(df['LEVEL_REGION'][df['TEAM'] == selected_team].drop_duplicates())[0]
    selected_state_cup_level = list(df['GOLD_CUP_LEVEL_REGION'][df['TEAM'] == selected_team].drop_duplicates())[0]

    #map values based on game type
    mapped_columns_df = {
    'FALL_LEAGUE':selected_region_level,
    'STATE_CUP': selected_state_cup_level,
    'ALL_GAMES' : 'ALL_GAMES'
    }

    #map columns based on game type
    mapped_columns_df_summary = {
    'FALL_LEAGUE':'LEVEL_REGION',
    'STATE_CUP': 'GOLD_CUP_LEVEL_REGION',
    'ALL_GAMES' : 'ALL_GAMES'}

    if selected_game_type == 'ALL_GAMES':
        wins_df = df_summary[((df_summary['LEVEL_REGION'] == mapped_columns_df['FALL_LEAGUE']) | (df_summary['GOLD_CUP_LEVEL_REGION'] == mapped_columns_df['STATE_CUP'])) & (df_summary['GAME_TYPE'] == 'ALL_GAMES') ].sort_values(by='WIN', ascending=False)

        pivot_data = df[
        ((df['LEVEL_REGION'] == selected_region_level ) & (df['GAME_TYPE'] == 'FALL_LEAGUE')) |
        ((df['GOLD_CUP_LEVEL_REGION'] == selected_state_cup_level ) & (df['GAME_TYPE'] == 'STATE_CUP'))
        ].groupby(['TEAM', 'OPPONENT'])['GAME_SCORE'] \
        .apply(lambda x: ', '.join(x.dropna()).strip(', ')) \
        .unstack(fill_value='') \
        .reset_index(drop=False)

        df_union_full_score_set = df_union_full_score_set[(df_union_full_score_set['GOLD_CUP_LEVEL_REGION'] == selected_state_cup_level) | (df_union_full_score_set['LEVEL_REGION'] == selected_region_level) ]

    else: 
        wins_df = df_summary[(df_summary[mapped_columns_df_summary[selected_game_type]] == mapped_columns_df[selected_game_type] ) & (df_summary['GAME_TYPE']==selected_game_type)].sort_values(by='WIN', ascending=False)

        pivot_data = df[
        (df[mapped_columns_df_summary[selected_game_type]] == mapped_columns_df[selected_game_type] ) &
        (df['GAME_TYPE'] == selected_game_type)
        ].groupby(['TEAM', 'OPPONENT'])['GAME_SCORE'] \
        .apply(lambda x: ', '.join(x.dropna()).strip(', ')) \
        .unstack(fill_value='') \
        .reset_index(drop=False)

        df_union_full_score_set = df_union_full_score_set[(df_union_full_score_set[mapped_columns_df_summary[selected_game_type]] == mapped_columns_df[selected_game_type]) & (df_union_full_score_set.GAME_TYPE ==  selected_game_type) ]
        df_union_full_score_set = df_union_full_score_set.drop_duplicates(subset='GAME_ID', keep='first')

    edges = []
    for i, row in  df_union_full_score_set.iterrows():
        edges.append(winner_loser(row['TEAM'],row['OPPONENT'],row['TEAM_SCORE'],row['OPPONENT_SCORE']))
    fig_data = plot_win_loss_graph(edges)


    wins_df = wins_df.drop(columns=['GAME_TYPE'])
    table_data = wins_df.to_dict('records')
    team_pivot_data = pivot_data.to_dict('records')
    return  table_data , team_pivot_data,  f'data:image/png;base64,{fig_data}'

@callback(
    Output('table-summary','data',allow_duplicate=True),
    Output('table-pivot','data',allow_duplicate=True),
    Output(component_id='bar-graph-matplotlib', component_property='src',allow_duplicate=True),
    Input('dropdown-LEVEL_REGION','value')
    #,prevent_initial_call='initial_duplicate'
    ,prevent_initial_call=True
	)


def level_region_dropdown(level_region):
    # Define the winner_loser function within level_region_dropdown
    # Filter dataframes based on the level_region
    wins_df = df_summary[
        (df_summary['LEVEL_REGION'] == level_region) & 
        (df_summary['GAME_TYPE'] == 'FALL_LEAGUE')
    ].sort_values(by='WIN', ascending=False)
    
    pivot_data = df[
        (df['LEVEL_REGION'] == level_region) &
        (df['GAME_TYPE'] == 'FALL_LEAGUE')
    ].groupby(['TEAM', 'OPPONENT'])['GAME_SCORE'] \
        .apply(lambda x: ', '.join(x.dropna()).strip(', ')) \
        .unstack(fill_value='') \
        .reset_index(drop=False)

    # Drop the GAME_TYPE column from wins_df
    wins_df = wins_df.drop(columns=['GAME_TYPE'])
    table_data = wins_df.to_dict('records')
    team_pivot_data = pivot_data.to_dict('records')

    # Process the data for graph plotting
    df_union_full_score_set = df[
        (df.TEAM_SCORE == df.TEAM_SCORE) & (df.TEAM_SCORE != df.OPPONENT_SCORE)
    ].copy()

    df_union_full_score_set = df_union_full_score_set[
        (df_union_full_score_set['LEVEL_REGION'] == level_region) & 
        (df_union_full_score_set.GAME_TYPE == 'FALL_LEAGUE')
    ]
    df_union_full_score_set = df_union_full_score_set.drop_duplicates(subset='GAME_ID', keep='first')

    edges = []
    for i, row in df_union_full_score_set.iterrows():
        edges.append(winner_loser(row['TEAM'], row['OPPONENT'], row['TEAM_SCORE'], row['OPPONENT_SCORE']))
    fig_data = plot_win_loss_graph(edges)

    return table_data, team_pivot_data, f'data:image/png;base64,{fig_data}'

@callback(
    Output('table-summary','data',allow_duplicate=True),
    Output('table-pivot','data',allow_duplicate=True),
    Output(component_id='bar-graph-matplotlib', component_property='src',allow_duplicate=True),
    Input('dropdown-STATE_CUP','value')
    #,prevent_initial_call='initial_duplicate'
    ,prevent_initial_call=True
    )

def level_region_dropdown_state_cup(level_state_cup):
    # Filter dataframes based on the level_region
    wins_df = df_summary[
        (df_summary['GOLD_CUP_LEVEL_REGION'] == level_state_cup) & 
        (df_summary['GAME_TYPE'] == 'STATE_CUP')
    ].sort_values(by='WIN', ascending=False)
    
    pivot_data = df[
        (df['GOLD_CUP_LEVEL_REGION'] == level_state_cup) &
        (df['GAME_TYPE'] == 'STATE_CUP')
    ].groupby(['TEAM', 'OPPONENT'])['GAME_SCORE'] \
        .apply(lambda x: ', '.join(x.dropna()).strip(', ')) \
        .unstack(fill_value='') \
        .reset_index(drop=False)

    # Drop the GAME_TYPE column from wins_df
    wins_df = wins_df.drop(columns=['GAME_TYPE'])
    table_data = wins_df.to_dict('records')
    team_pivot_data = pivot_data.to_dict('records')

    # Process the data for graph plotting
    df_union_full_score_set = df[
        (df.TEAM_SCORE == df.TEAM_SCORE) & (df.TEAM_SCORE != df.OPPONENT_SCORE)
    ].copy()

    df_union_full_score_set = df_union_full_score_set[
        (df_union_full_score_set['GOLD_CUP_LEVEL_REGION'] == level_state_cup) & 
        (df_union_full_score_set.GAME_TYPE == 'STATE_CUP')
    ]
    df_union_full_score_set = df_union_full_score_set.drop_duplicates(subset='GAME_ID', keep='first')

    edges = []
    for i, row in df_union_full_score_set.iterrows():
        edges.append(winner_loser(row['TEAM'], row['OPPONENT'], row['TEAM_SCORE'], row['OPPONENT_SCORE']))
    fig_data = plot_win_loss_graph(edges)
  

    return table_data, team_pivot_data, f'data:image/png;base64,{fig_data}'

#team schedule
@app.callback(
    Output('schedule-table', 'data'),
    Input('dropdown-team', 'value')
)
def update_table(selected_team_one):
    filtered_df = schedule_df

    if selected_team_one:
        filtered_df = filtered_df[(filtered_df['TEAM_ONE'] == selected_team_one) | (filtered_df['TEAM_TWO'] == selected_team_one)]
    
    return filtered_df.to_dict('records')



if __name__ == '__main__':
    app.run(debug=True)