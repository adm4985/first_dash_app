from dash import Dash, html, dcc, callback, Output, Input, dash_table, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import io
import base64
matplotlib.use('agg')

# Data processing
schedule_df = pd.read_csv('full_schedule.csv')[['DATETIME', 'TEAM_ONE', 'SCORE', 'TEAM_TWO', 'LOCATION', 'GAME_TYPE']]
schedule_df.sort_values(by='DATETIME', inplace=True)

df = pd.read_csv('norCalResults.csv').sort_values(by='TEAM')
df_summary_gt = df.groupby(['LEVEL_REGION', 'GOLD_CUP_LEVEL_REGION', 'GAME_TYPE', 'TEAM']).agg({
    'WIN': 'sum', 'LOSS': 'sum', 'DRAW': 'sum', 'TEAM_SCORE': 'sum', 'OPPONENT_SCORE': 'sum', 'GOAL_DIFF': 'sum'
}).reset_index()

df_summary_all = df.groupby(['LEVEL_REGION', 'GOLD_CUP_LEVEL_REGION', 'TEAM']).agg({
    'WIN': 'sum', 'LOSS': 'sum', 'DRAW': 'sum', 'TEAM_SCORE': 'sum', 'OPPONENT_SCORE': 'sum', 'GOAL_DIFF': 'sum'
}).reset_index()

df_summary = pd.concat([df_summary_gt, df_summary_all], ignore_index=True)
df_summary['MP'] = df_summary['WIN'] + df_summary['LOSS'] + df_summary['DRAW']
df_summary['POINTS'] = df_summary['WIN'] * 3 + df_summary['DRAW']
df_summary = df_summary[['LEVEL_REGION', 'GOLD_CUP_LEVEL_REGION', 'GAME_TYPE', 'TEAM', 'MP', 'WIN', 'LOSS', 'DRAW',
                         'TEAM_SCORE', 'OPPONENT_SCORE', 'GOAL_DIFF', 'POINTS']].sort_values(
    by=['WIN', 'POINTS', 'GOAL_DIFF', 'LOSS'], ascending=[False, False, True, True])
df_summary['GAME_TYPE'] = df_summary['GAME_TYPE'].fillna('ALL_GAMES')


gt = sorted(df_summary.GAME_TYPE.unique())

# App setup
app = Dash(external_stylesheets=[dbc.themes.DARKLY])

# Helper function for dropdown
def create_dropdown(id_name, label, options, default_value):
    return html.Div([
        html.H3(label),
        dcc.Dropdown(
            id=id_name,
            options=[{'label': opt, 'value': opt} for opt in options if opt == opt],
            value=default_value,
            style={'width': '100%', 'border': '1px solid #ccc', 'borderRadius': '5px', 'backgroundColor': 'white',
                   'color': 'black', 'fontSize': '16px'}
        )
    ], style={'padding': '10px', 'flex': '1', 'boxSizing': 'border-box'})

# Layout
app.layout = html.Div([
    html.H1('Results', style={'textAlign': 'center'}),
    html.Div([
        create_dropdown('dropdown-team', 'Select Team', df.TEAM.unique(),
                        'Castro Valley Soccer Club Castro Valley SC CVSC United Green 2015 G'),
        create_dropdown('dropdown-game_type', 'Select Game Type', gt, 'FALL_LEAGUE'),
        create_dropdown('dropdown-LEVEL_REGION', 'Select Level Region',
                        df_summary[df_summary.GAME_TYPE == 'FALL_LEAGUE'].LEVEL_REGION.unique(), None),
        create_dropdown('dropdown-STATE_CUP', 'Select State Cup',
                        df_summary[df_summary.GAME_TYPE == 'STATE_CUP'].GOLD_CUP_LEVEL_REGION.unique(), None)
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between', 'gap': '10px'}),
    html.Hr(style={'border': '1px solid #ccc', 'margin': '20px 0'}),
    html.Div([
        dash_table.DataTable(id='table-summary', fill_width=False, style_header={
            'backgroundColor': 'gray', 'textAlign': 'center'}, style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(105, 105, 105)'}], style_table={'width': '100%', 'maxWidth': '1500px', 'overflowX': 'auto'}),
        html.Img(id='bar-graph-matplotlib'),
        dash_table.DataTable(id='table-pivot', style_header={'backgroundColor': 'gray', 'textAlign': 'center'},
                             style_table={'width': '80%', 'maxWidth': '1500px', 'overflowX': 'auto'}),
        dash_table.DataTable(id='schedule-table', columns=[{"name": i, "id": i} for i in schedule_df.columns],
                             data=schedule_df.to_dict('records'), style_table={'overflowX': 'auto', 'maxWidth': '1500px', 'height': '300px', 'overflowY': 'auto'})
    ])
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

# Callback function
@callback(
    Output('table-summary', 'data'),
    Output('table-pivot', 'data'),
    Output('bar-graph-matplotlib', 'src'),
    [Input('dropdown-team', 'value'), Input('dropdown-game_type', 'value'), Input('dropdown-LEVEL_REGION', 'value'), Input('dropdown-STATE_CUP', 'value')]
)
def update_tables(selected_team, selected_game_type, level_region, state_cup):
    
    #set team and gold cup regions for the selected team
    selected_region_level = df['LEVEL_REGION'][df['TEAM'] == selected_team].drop_duplicates().values[0]
    selected_state_cup_level = df['GOLD_CUP_LEVEL_REGION'][df['TEAM'] == selected_team].drop_duplicates().values[0]

    df_union_full_score_set = df[(df.TEAM_SCORE == df.TEAM_SCORE) & (df.TEAM_SCORE != df.OPPONENT_SCORE)].copy()

    #map values based on game type
    mapped_columns_df = {    'FALL_LEAGUE':selected_region_level,    'STATE_CUP': selected_state_cup_level,    'ALL_GAMES' : 'ALL_GAMES'    }

    #map columns based on game type
    mapped_columns_df_summary = {    'FALL_LEAGUE':'LEVEL_REGION',    'STATE_CUP': 'GOLD_CUP_LEVEL_REGION',    'ALL_GAMES' : 'ALL_GAMES'}

    triggered = callback_context.triggered_id
    
    if triggered == 'dropdown-team' or triggered == 'dropdown-game_type':
        if selected_game_type == 'ALL_GAMES':
            wins_df = df_summary[((df_summary['LEVEL_REGION'] == mapped_columns_df['FALL_LEAGUE']) | (df_summary['GOLD_CUP_LEVEL_REGION'] == mapped_columns_df['STATE_CUP'])) & (df_summary['GAME_TYPE'] == selected_game_type) ].sort_values(by='WIN', ascending=False)

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


    edges = [(row['TEAM'], row['OPPONENT']) for _, row in df_union_full_score_set.iterrows()]





    fig_data = plot_win_loss_graph(edges)
    table_data = df_summary[df_summary.GAME_TYPE == selected_game_type].to_dict('records')
    pivot_data = pivot_data.to_dict('records')

    return table_data, pivot_data, f'data:image/png;base64,{fig_data}'

if __name__ == '__main__':
    app.run_server(debug=True)
