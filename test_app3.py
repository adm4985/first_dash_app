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

    

    
    if selected_game_type == 'FALL_LEAGUE':
        wins_df = df_summary[(df_summary['LEVEL_REGION'] == selected_region_level ) & (df_summary['GAME_TYPE']==selected_game_type)].sort_values(by='WIN', ascending=False)
        
        pivot_data = df[
            (df['LEVEL_REGION'] == selected_region_level) &
            (df['GAME_TYPE'] == 'FALL_LEAGUE')
        ].groupby(['TEAM', 'OPPONENT'])['GAME_SCORE'] \
            .apply(lambda x: ', '.join(x.dropna()).strip(', ')) \
            .unstack(fill_value='') \
            .reset_index(drop=False)
    
        for i, row in wins_df.iterrows():
            print(row['TEAM'])
    if selected_game_type == 'STATE_CUP':
        wins_df = df_summary[(df_summary['LEVEL_REGION'] == selected_state_cup_level ) & (df_summary['GAME_TYPE']==selected_game_type)].sort_values(by='WIN', ascending=False)

        pivot_data = df[
            (df['GOLD_CUP_LEVEL_REGION'] == selected_state_cup_level) &
            (df['GAME_TYPE'] == 'STATE_CUP')
        ].groupby(['TEAM', 'OPPONENT'])['GAME_SCORE'] \
            .apply(lambda x: ', '.join(x.dropna()).strip(', ')) \
            .unstack(fill_value='') \
            .reset_index(drop=False)
    
    else:
        print('no data')

    wins_df = wins_df.drop(columns=['GAME_TYPE'])
    table_data = wins_df.to_dict('records')
    team_pivot_data = pivot_data.to_dict('records')
    return  table_data , team_pivot_data

@callback(
    Output('table-summary','data',allow_duplicate=True),
    Output('table-pivot','data',allow_duplicate=True),
    Input('dropdown-LEVEL_REGION','value'),
    prevent_initial_call='initial_duplicate'
	)

def level_region_dropdown(level_region):

	wins_df = df_summary[(df_summary['LEVEL_REGION'] == level_region ) & (df_summary['GAME_TYPE']=='FALL_LEAGUE')].sort_values(by='WIN', ascending=False)
	pivot_data = df[
		(df['LEVEL_REGION'] == level_region) &
		(df['GAME_TYPE'] == 'FALL_LEAGUE')
		].groupby(['TEAM', 'OPPONENT'])['GAME_SCORE'] \
		.apply(lambda x: ', '.join(x.dropna()).strip(', ')) \
		.unstack(fill_value='') \
		.reset_index(drop=False)

	wins_df = wins_df.drop(columns=['GAME_TYPE'])
	table_data = wins_df.to_dict('records')
	team_pivot_data = pivot_data.to_dict('records') 

	return table_data , team_pivot_data

@callback(
    Output(component_id='bar-graph-matplotlib', component_property='src'),
    Input('dropdown-team','value')
    )

def update_team_dropdown(selected_team):
    #https://plotly.com/blog/dash-matplotlib/
    def winner_loser(team1, team2, score1, score2):
        if score1 > score2:
            winner = team1
            loser = team2
        elif score2 > score1:
            winner = team2
            loser = team1
        return winner, loser


    selected_region_level = list(df['LEVEL_REGION'][df['TEAM'] == selected_team].drop_duplicates())[0]

    df_union_full_score_set = df[(df.TEAM_SCORE == df.TEAM_SCORE) & (df.TEAM_SCORE != df.OPPONENT_SCORE)].copy()
    df_union_full_score_set = df_union_full_score_set[(df_union_full_score_set.LEVEL_REGION == selected_region_level) & (df_union_full_score_set.GAME_TYPE == 'FALL_LEAGUE') ]
    df_union_full_score_set = df_union_full_score_set.drop_duplicates(subset='GAME_ID', keep='first') 

    edges = []
    for i, row in  df_union_full_score_set.iterrows():
        edges.append(winner_loser(row['TEAM'],row['OPPONENT'],row['TEAM_SCORE'],row['OPPONENT_SCORE']) )


# Clear any previous figure
    plt.clf()
    plt.figure(figsize=(10, 8))
    plt.title(f'{selected_region_level} Win-Loss')
    # Build the matplotlib figure
    G = nx.DiGraph()
    #edges = [('B', 'A'), ('C', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('C', 'B'), ('A', 'C'), ('B', 'C')]
    G.add_edges_from(edges)

    # Draw the graph
    pos = nx.spring_layout(G, k=3, iterations=100)  # Increase k to spread nodes apart
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=750, edge_color='darkblue', font_size=8, arrows=True)
    

    # Save it to a temporary buffer.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)  # Move to the beginning of the buffer

    # Embed the result in the HTML output.
    fig_data = base64.b64encode(buf.getvalue()).decode("ascii")
    buf.close()  # Close the buffer

    return f'data:image/png;base64,{fig_data}'

if __name__ == '__main__':
    app.run(debug=True)