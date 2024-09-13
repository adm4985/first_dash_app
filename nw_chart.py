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

#process data and create dataframes
df = pd.read_csv('norCalResults.csv')
def winner_loser(team1, team2, score1, score2):
    if score1 > score2:
        winner = team1
        loser = team2
    elif score2 > score1:
        winner = team2
        loser = team1
    return winner, loser

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

    html.Img(id='bar-graph-matplotlib')

    ]

@callback(
    Output(component_id='bar-graph-matplotlib', component_property='src'),
    Input('dropdown-team','value')
    )

def update_team_dropdown(selected_team):
#https://plotly.com/blog/dash-matplotlib/
# Clear any previous figure
    selected_region_level = list(df['LEVEL_REGION'][df['TEAM'] == selected_team].drop_duplicates())[0]
    print(selected_region_level)
    df_union_full_score_set = df[(df.TEAM_SCORE == df.TEAM_SCORE) & (df.TEAM_SCORE != df.OPPONENT_SCORE)]
    df_union_full_score_set = df_union_full_score_set[(df_union_full_score_set.LEVEL_REGION == selected_region_level) & (df_union_full_score_set.GAME_TYPE == 'FALL_LEAGUE') ]
    df_union_full_score_set = df_union_full_score_set.drop_duplicates(subset='GAME_ID', keep='first') 

    edges = []
    for i, row in  df_union_full_score_set.iterrows():
        edges.append(winner_loser(row['TEAM'],row['OPPONENT'],row['TEAM_SCORE'],row['OPPONENT_SCORE']) )



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
    app.run_server(debug=True)
    #http://10.0.0.37:5000/