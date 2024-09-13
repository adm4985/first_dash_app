# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 19:31:29 2024

@author: alexm
"""

# Define browser headers
# =============================================================================
# headers = {
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36',
#     'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
#     'Accept-Language': 'en-US,en;q=0.9',
#     'Accept-Encoding': 'gzip, deflate, br',
#     'Connection': 'keep-alive',
#     'Upgrade-Insecure-Requests': '1',
#     'TE': 'Trailers'
# }
# =============================================================================


#State Cup Results=============================================================================
url = "https://system.gotsport.com/org_event/events/33460/schedules?group=277140"
page = session.get(url)
soup = BeautifulSoup(page.content, "html.parser")

data = []

    
table = soup.find_all('table', attrs={'class':'table table-bordered table-condensed table-hover'})
 
#data = []
for t in table: 
    table_body = t.find('tbody')
    rows = table_body.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        #data.append([ele for ele in cols if ele]) # Get rid of empty values
        data.append([ele for ele in cols]) # Get rid of empty values  

df_sc = pd.DataFrame(data)
columns = ['GAME_ID','DATE','TEAM_ONE','SCORE','TEAM_TWO','LOCATION','STATE_CUP_TIER']
df_sc.columns = columns
df_sc['GAME_TYPE'] = 'STATE_CUP'
#=============================================================================

from requests_html import HTMLSession
from bs4 import BeautifulSoup
import pandas as pd

session = HTMLSession()


#State Cup Results=============================================================================
url_list = ["https://system.gotsport.com/org_event/events/33460/schedules?date=All&group=277140",
            "https://system.gotsport.com/org_event/events/33460/schedules?date=All&group=277138",
            "https://system.gotsport.com/org_event/events/33460/schedules?date=All&group=277139"]

sc_data = []

for url in url_list:
    page = session.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    
    panel = soup.find_all('div', attrs={'class':'panel-group'})   

    for p in panel:
        t = str(p)
        try:    
            soup = BeautifulSoup(t, 'html.parser')
            
            # Find the group section
            panel_group = soup.find('div', class_='panel-group')
            
            # Extract the group name and date
            panel_heading = panel_group.find('div', class_='panel-heading')
            group_info = panel_heading.find('a').get_text(strip=True).split('|')
            group_name = group_info[0].strip()
            group_date = group_info[1].strip()
            
            # Find the table
            table = panel_group.find('table')
            
            # Extract table headers
            headers = [header.get_text(strip=True) for header in table.find_all('th')]
            
            # Extract table rows
            rows = []
            for row in table.find_all('tr')[1:]:  # Skip header row
                cols = [col.get_text(strip=True) for col in row.find_all('td')]
                cols.append(group_name)
                rows.append(cols)
                sc_data.extend(rows)
    
        except IndexError:
            print("error")

#Clean Up Data
df_sc = pd.DataFrame(sc_data)
df_sc = df_sc.drop_duplicates().reset_index(drop=True)
columns = ['GAME_ID','DATE','TEAM_ONE','SCORE','TEAM_TWO','LOCATION','STATE_CUP_TIER','STATE_CUP_GROUP']
df_sc.columns = columns
df_sc['TEAM_ONE'] = df_sc['TEAM_ONE'].astype(str)
df_sc['GAME_TYPE'] = 'STATE_CUP'
df_sc.to_csv(r'C:\Users\alexm\dash\df.csv')
#=============================================================================

######GET LEAGUE IDs############
gotsporturl = "https://system.gotsport.com/org_event/events/33458"

page = session.get(gotsporturl)
soup = BeautifulSoup(page.content, "html.parser")
divs =  soup.find_all('div', attrs={'class':'age-group col-md-12 group-u10'}) #get tiers for age U10

links_list = []
for d in divs:
    link_body = d.find_all('a')
    links_list = [link.get('href') for link in link_body]

tier_ids = [s[-6:] for s in links_list]

tier_ids = list(set(tier_ids))
####################################
 


#######GET FALL LEAGUE STATS#################
url = "https://system.gotsport.com/org_event/events/33458/schedules?date=All&group="
fl_data = []
for tier in tier_ids:
    url_tier = url+tier

    
    page = session.get(url_tier)
    soup = BeautifulSoup(page.content, "html.parser")
    
    table = soup.find_all('table', attrs={'class':'table table-bordered table-condensed table-hover'})
     
    #data = []
    for t in table: 
        table_body = t.find('tbody')
        rows = table_body.find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            cols = [ele.text.strip() for ele in cols]
            #data.append([ele for ele in cols if ele]) # Get rid of empty values
            fl_data.append([ele for ele in cols]) # Get rid of empty values  

df = pd.DataFrame(fl_data)
df.to_csv(r'C:\Users\alexm\dash\df.csv')
columns = ['GAME_ID','DATE','TEAM_ONE','SCORE','TEAM_TWO','LOCATION','LEVEL_REGION']
df.columns = columns
df['TEAM_ONE'] = df['TEAM_ONE'].astype(str)
#df['STATE_CUP_GROUP']=None
df['GAME_TYPE'] = 'FALL_LEAGUE'
#######################################################################################################################


df['TEAM_GSB_LEVEL'] = df['LEVEL_REGION'].apply(lambda x: x.split()[0])
df['REGION'] = df['LEVEL_REGION'].apply(lambda x: ' '.join(x.split()[1:]).replace('-','').lstrip())

#create team detail tables
df_agg = df[['TEAM_ONE','TEAM_GSB_LEVEL','REGION']].drop_duplicates().copy()
df.drop(['LEVEL_REGION','TEAM_GSB_LEVEL','REGION'], axis=1, inplace=True)


df_sc_agg =  pd.DataFrame(df_sc[['TEAM_ONE','STATE_CUP_GROUP','STATE_CUP_TIER']].drop_duplicates()).copy()
df_sc.drop(['STATE_CUP_GROUP','STATE_CUP_TIER'], axis=1, inplace=True)
 
####################################

df_combined = pd.concat([df,df_sc]).copy()

 
df_combined = df_combined.query("TEAM_ONE != '' ")



df_combined = pd.merge(df_combined, df_sc_agg,how='left',left_on='TEAM_ONE',right_on='TEAM_ONE')
df_combined = pd.merge(df_combined, df_agg,how='left',left_on='TEAM_ONE',right_on='TEAM_ONE')

df_combined['TEAM_ONE_SCORE'] = df_combined['SCORE'].apply(lambda x:  int(x.split('-')[0]) if len(x) > 2  else None )
df_combined['TEAM_TWO_SCORE'] = df_combined['SCORE'].apply(lambda x:  int(x.split('-')[1]) if len(x) > 2  else None )


 
df_combined['DATETIME'] = pd.to_datetime(df_combined['DATE'].str[:12], format='%b %d, %Y') 
df_combined['DATETIME'] = df_combined['DATETIME'].apply(lambda x: x.date())
 
df_union = df_combined[['GAME_ID','DATETIME','TEAM_ONE','TEAM_ONE_SCORE','TEAM_TWO','TEAM_TWO_SCORE','LOCATION','GAME_TYPE']].copy()
df_union.columns = ['GAME_ID','DATE','TEAM','TEAM_SCORE','OPPONENT','OPPONENT_SCORE','LOCATION','GAME_TYPE']
df_union['HOME_AWAY'] = 'Home'

df_union2 = df_combined[['GAME_ID','DATETIME','TEAM_TWO','TEAM_TWO_SCORE','TEAM_ONE','TEAM_ONE_SCORE','LOCATION','GAME_TYPE']].copy()
df_union2.columns = ['GAME_ID','DATE','TEAM','TEAM_SCORE','OPPONENT','OPPONENT_SCORE','LOCATION','GAME_TYPE']
df_union2['HOME_AWAY'] = 'Away'
df_union_full = pd.concat([df_union,df_union2])

df_union_full['WIN'] = df_union_full.apply(lambda row: 1 if row['TEAM_SCORE'] > row['OPPONENT_SCORE'] else None, axis=1)
df_union_full['LOSS'] = df_union_full.apply(lambda row: 1 if row['TEAM_SCORE'] < row['OPPONENT_SCORE'] else None, axis=1)
df_union_full['DRAW'] = df_union_full.apply(lambda row: 1 if row['TEAM_SCORE'] == row['OPPONENT_SCORE'] else None, axis=1)
df_union_full['GOAL_DIFF'] = df_union_full['TEAM_SCORE'] - df_union_full['OPPONENT_SCORE']
df_union_full['GAME_SCORE'] = df_union_full.apply(lambda row: "{}-{}".format(int(row['TEAM_SCORE']), int(row['OPPONENT_SCORE'])) if row['TEAM_SCORE']==row['TEAM_SCORE'] else None ,axis=1)

df_union_full = pd.merge(df_union_full, df_sc_agg,how='left',left_on='TEAM',right_on='TEAM_ONE')
df_union_full = pd.merge(df_union_full, df_agg,how='left',left_on='TEAM',right_on='TEAM_ONE')
df_union_full.drop(['TEAM_ONE_x','TEAM_ONE_y'], axis=1, inplace=True)

df_union_full = pd.merge(df_union_full, df_sc_agg,how='left',left_on='OPPONENT',right_on='TEAM_ONE')
df_union_full = pd.merge(df_union_full, df_agg,how='left',left_on='OPPONENT',right_on='TEAM_ONE')
df_union_full.drop(['TEAM_ONE_x','TEAM_ONE_y'], axis=1, inplace=True)

f_columns = ['GAME_ID', 'DATE', 'TEAM', 'TEAM_SCORE', 'OPPONENT', 'OPPONENT_SCORE',
       'LOCATION', 'GAME_TYPE', 'HOME_AWAY', 'WIN', 'LOSS', 'DRAW',
       'GOAL_DIFF', 'GAME_SCORE', 'TEAM_STATE_CUP_GROUP', 'TEAM_STATE_CUP_TIER',
       'TEAM_GSB_LEVEL', 'TEAM_REGION', 'OPPONENT_STATE_CUP_GROUP', 'OPPONENT_STATE_CUP_TIER',
       'OPPONENT_TEAM_GSB_LEVEL', 'OPPONENT_REGION']

df_union_full.columns = f_columns
df_union_full['LEVEL_REGION'] = df_union_full['TEAM_GSB_LEVEL']+'-'+df_union_full['TEAM_REGION']
df_union_full['GOLD_CUP_LEVEL_REGION'] = df_union_full['TEAM_STATE_CUP_TIER']+'-'+df_union_full['TEAM_STATE_CUP_GROUP']

df_union_full['GOLD_CUP_LEVEL_REGION'] = df_union_full['GOLD_CUP_LEVEL_REGION'].fillna('')
df_union_full['LEVEL_REGION'] = df_union_full['LEVEL_REGION'].fillna('')

df_union_full.to_csv(r'C:\Users\alexm\dash\first_dash_app\norCalResults.csv')

####################################
df_union_full = pd.read_csv(r'C:\Users\alexm\dash\first_dash_app\norCalResults.csv',header=0)

team_pivot = df_union_full[(df_union_full['LEVEL_REGION']=='Silver-Region 2') & (df_union_full['GAME_TYPE']=='FALL_LEAGUE')].groupby(['TEAM', 'OPPONENT'])['GAME_SCORE'].apply(lambda x: ', '.join(x.dropna()).strip(', ')).unstack(fill_value='').reset_index(drop=False)

TEST = df_union_full[['DATE','TEAM','OPPONENT','TEAM_SCORE','GAME_SCORE']][df_union_full['TEAM']=='Castro Valley Soccer Club Castro Valley SC CVSC United 2015 Girls Yellow'].sort_values(by='DATE')
 



####################################
l = sorted(df.TEAM_ONE.unique())

f =  list(df_union_full['LEVEL_REGION'][df_union_full['TEAM']=='Castro Valley Soccer Club Castro Valley SC CVSC United 2015 Girls Yellow'].drop_duplicates())[0]
list(f)[0]
              
df_union_full.dtypes
t = df_union_full['LEVEL_REGION'][df_union_full['TEAM']=='Castro Valley Soccer Club Castro Valley SC CVSC United 2015 Girls Yellow'].drop_duplicates()
t = df_union_full[df_union_full['LEVEL_REGION']=='Silver - Region 4'].copy()
t['GAME_SCORE'] = t.apply(lambda row: "{}-{}".format(int(row['TEAM_SCORE']), int(row['OPPONENT_SCORE'])) if row['TEAM_SCORE']==row['TEAM_SCORE'] else '' ,axis=1)

t.sort_values(by='WIN', ascending=False)

grouped = t.groupby(['TEAM', 'OPPONENT'])['GAME_SCORE'].apply(lambda x: ', '.join(x)  ).unstack(fill_value='')
grouped = t[t['LEVEL_REGION']=='Silver - Region 4'].groupby(['TEAM', 'OPPONENT'])['GAME_SCORE'].apply(lambda x: ', '.join(x.dropna()).strip(', ')).unstack(fill_value='').reset_index(drop=False)


pivot_df = pd.pivot_table(
    t,
    values='GAME_SCORE',
    index='TEAM',
    columns='OPPONENT',
    aggfunc=lambda x: ', '.join(x),  # Aggregate by joining results as strings
       # Fill NaN values with 0
)

"{}-{}".format(int(row['TEAM_SCORE']), int(row['TEAM_SCORE']))
df_union_full['TEAM_SCORE'] = df_union_full['TEAM_SCORE'].astype(int, errors='ignore')
df_union_full['OPPONENT_SCORE'] = df_union_full['OPPONENT_SCORE'].astype(int, errors='ignore')

t = df_union_full.groupby(['LEVEL_REGION','TEAM'])['WIN','LOSS','DRAW','TEAM_SCORE'].sum().reset_index(drop=False)

df_summary_fl = df_union_full[df_union_full.GAME_TYPE=='FALL_LEAGUE'].groupby(['LEVEL_REGION', 'GOLD_CUP_LEVEL_REGION', 'GAME_TYPE','TEAM']).agg({'WIN':'sum','LOSS':'sum','DRAW':'sum','TEAM_SCORE':'sum','OPPONENT_SCORE':'sum','GOAL_DIFF':'sum'}).reset_index(drop=False)
df_summary_sc = df_union_full[df_union_full.GAME_TYPE=='STATE_CUP'].groupby(['GOLD_CUP_LEVEL_REGION','LEVEL_REGION','GAME_TYPE','TEAM']).agg({'WIN':'sum','LOSS':'sum','DRAW':'sum','TEAM_SCORE':'sum','OPPONENT_SCORE':'sum','GOAL_DIFF':'sum'}).reset_index(drop=False)
df_summary_sc = df_summary_sc.rename(columns={'GOLD_CUP_LEVEL_REGION': 'LEVEL_REGION','LEVEL_REGION':'GOLD_CUP_LEVEL_REGION'})
df_summary = pd.concat([df_summary_fl, df_summary_sc], ignore_index=True)

df_summary['MP'] = df_summary['WIN']+df_summary['LOSS']+df_summary['DRAW']
df_summary['POINTS'] = df_summary['WIN'] * 3 + df_summary['DRAW']
df_summary = df_summary[['LEVEL_REGION','GOLD_CUP_LEVEL_REGION','GAME_TYPE','TEAM','MP','WIN','LOSS','DRAW','TEAM_SCORE','OPPONENT_SCORE','GOAL_DIFF','POINTS']]
df_summary.sort_values(by='WIN', ascending=False)

df_region_list = sorted(df_summary[df_summary.GAME_TYPE=='FALL_LEAGUE']['LEVEL_REGION'].drop_duplicates())
df_region_list.sort_values(by='LEVEL_REGION', ascending=True)


for i in df_region_list:
    print(i)

for i in df_summary[df_summary.GAME_TYPE=='FALL_LEAGUE'].sort_values(by='LEVEL_REGION', ascending=False).LEVEL_REGION.unique():
    print(i)

pivot_data = df_union_full[
            (df_union_full['LEVEL_REGION'] == 'U10 Girls Gold/Silver-Group B') &
            (df_union_full['GAME_TYPE'] == 'STATE_CUP')
        ].groupby(['TEAM', 'OPPONENT'])['GAME_SCORE'] \
            .apply(lambda x: ', '.join(x.dropna()).strip(', ')) \
            .unstack(fill_value='') \
            .reset_index(drop=False)

df_union_full.groupby('TEAM')['WIN','LOSS'].sum().reset_index(drop=False)

def outcome(score1,score2):
    result = ''
    if score1:
        if score1 > score2:
            result ='W'
        if score1 < score2:
            result ='L'
        if score1 == score2:
            result ='D'
    return result

df_union_full['OUTCOME'] = df_union_full.apply(lambda row: outcome(row['TEAM_SCORE'],row['OPPONENT_SCORE']),axis=1)
df_union_full['WIN'] = df_union_full.apply(lambda row: 1 if row['TEAM_SCORE'] > row['OPPONENT_SCORE'] else 0, axis=1)
df_union_full['LOSS'] = df_union_full.apply(lambda row: 1 if row['TEAM_SCORE'] < row['OPPONENT_SCORE'] else 0, axis=1)
df_union_full['DRAW'] = df_union_full.apply(lambda row: 1 if row['TEAM_SCORE'] == row['OPPONENT_SCORE'] else 0, axis=1)

df_union_full.groupby(['LEVEL'])['GAME_ID','TEAM'].count()
result_table = df_union_full.groupby(['TEAM'])['WIN','LOSS','DRAW','TEAM_SCORE','OPPONENT_SCORE'].sum().reset_index(drop=False)
df_union_full[df_union_full['TEAM']].query("LEVEL == 'Silver - Region 4'")


data = {
    'HomeTeam': ['A', 'A', 'C', 'B', 'B', 'C'],
    'AwayTeam': ['B', 'C', 'B', 'A', 'C', 'A'],
    'Points': [3, 2, 4, 2, 3, 1]
}