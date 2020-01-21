
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import math



"""college player stats"""

def dataframe(url):
    stats = requests.get(url)
    soup = BeautifulSoup(stats.content, 'html.parser')
    table = soup.find(name='table',attrs={'id':'stats'})
   
    html_str = str(table)
    df = pd.read_html(html_str)[0]
    
    renamed = []
    for column in df.columns.values:
        renamed.append(column[1])
    df.columns = renamed
    df['Rk'].fillna(value='Rk',inplace=True)
    df = pd.DataFrame(df.loc[df['Rk'] != 'Rk'])
    return df

#create empty dataframes to fill
college_centers = pd.DataFrame()
college_guards = pd.DataFrame()
college_forwards = pd.DataFrame()


for i in list(np.arange(0,50000,100)):
    try:
        url = "https://www.sports-reference.com/cbb/play-index/psl_finder.cgi?request=1&match=combined&year_min=1993&year_max=2020&conf_id=&school_id=&class_is_fr=Y&class_is_so=Y&class_is_jr=Y&class_is_sr=Y&pos_is_cf=Y&pos_is_c=Y&games_type=A&qual=&c1stat=&c1comp=&c1val=&c2stat=&c2comp=&c2val=&c3stat=&c3comp=&c3val=&c4stat=&c4comp=&c4val=&order_by=pts&order_by_asc=&offset={}".format(i)
        df = dataframe(url)
        college_centers = college_centers.append(df)
        #for verbosity purposes
        print('record #:', i)
    except ValueError:
        print('No more rows')
        break


for i in list(np.arange(0,50000,100)):
    try:
        url = "https://www.sports-reference.com/cbb/play-index/psl_finder.cgi?request=1&match=combined&year_min=1993&year_max=2020&conf_id=&school_id=&class_is_fr=Y&class_is_so=Y&class_is_jr=Y&class_is_sr=Y&pos_is_g=Y&pos_is_gf=Y&games_type=A&qual=&c1stat=&c1comp=&c1val=&c2stat=&c2comp=&c2val=&c3stat=&c3comp=&c3val=&c4stat=&c4comp=&c4val=&order_by=pts&order_by_asc=&offset={}".format(i)
        df = dataframe(url)
        college_guards = college_guards.append(df)
        print('record #:', i)
    except ValueError:
        print('No more rows')
        break

for i in list(np.arange(0,50000,100)):
    try:
        url = "https://www.sports-reference.com/cbb/play-index/psl_finder.cgi?request=1&match=combined&year_min=1993&year_max=2020&conf_id=&school_id=&class_is_fr=Y&class_is_so=Y&class_is_jr=Y&class_is_sr=Y&pos_is_f=Y&pos_is_fc=Y&games_type=A&qual=&c1stat=&c1comp=&c1val=&c2stat=&c2comp=&c2val=&c3stat=&c3comp=&c3val=&c4stat=&c4comp=&c4val=&order_by=pts&order_by_asc=&offset={}".format(i)
        df = dataframe(url)
        college_forwards = college_forwards.append(df)
        print('record #:', i)
    except ValueError:
        print('No more rows')
        break
    

college_centers['Position'] = 'C'
college_guards['Position'] = 'G'    
college_forwards['Position'] = 'F' 

#make final frame with all positions
college_data = pd.concat([college_centers,college_guards,college_forwards])       
college_data.fillna(value=0,inplace=True)  

#filter out fake Kevin Johnsons and Shawn Kemps
college_data = pd.DataFrame(college_data.loc[(college_data['Player'] != 'Kevin Johnson') &
                                (college_data['Player'] != 'Shawn Kemp') &
                                (college_data['Player'] != 'Tony Parker')]) 
college_data.reset_index(drop=True,inplace=True)

#clean up dtypes
for column in college_data.columns:
    if column in ['Player','School','Conf','Position']:
        college_data[column] = college_data[column].astype(str)
    else:
        college_data[column] = college_data[column].astype(float)

"""NBA League Average Stats"""
url = "https://www.basketball-reference.com/leagues/NBA_stats_per_game.html"
nba_averages_by_year = dataframe(url)

for column in nba_averages_by_year.columns:
    if column in ['Season','Lg','Ht']:
        nba_averages_by_year[column] = nba_averages_by_year[column].astype(str)
    else:
        nba_averages_by_year[column] = nba_averages_by_year[column].astype(float)

seasons = []
for season in nba_averages_by_year['Season']:
    seasons.append(int(season.split("-")[0]) + 1)

nba_averages_by_year['Season'] = seasons    
nba_averages_by_year['TS%'] = nba_averages_by_year['PTS'] / (2 * (nba_averages_by_year['FGA'] + (nba_averages_by_year['FTA'] * 0.44)))   
nba_averages_by_year['Pts/Poss'] = nba_averages_by_year['PTS'] / nba_averages_by_year['Pace']


"""NBA player stats"""


#need to make this a loop similar to above college data
url = "https://www.basketball-reference.com/play-index/psl_finder.cgi?request=1&match=combined&type=totals&per_minute_base=36&per_poss_base=100&lg_id=NBA&is_playoffs=N&year_min=1994&year_max=2020&franch_id=&season_start=1&season_end=-1&age_min=0&age_max=99&shoot_hand=&height_min=0&height_max=99&birth_country_is=Y&birth_country=&birth_state=&college_id=&draft_year=&is_active=&debut_yr_nba_start=&debut_yr_nba_end=&is_hof=&is_as=&as_comp=gt&as_val=1&award=&pos_is_g=Y&pos_is_gf=Y&pos_is_f=Y&pos_is_fg=Y&pos_is_fc=Y&pos_is_c=Y&pos_is_cf=Y&qual=&c1stat=&c1comp=&c1val=&c2stat=&c2comp=&c2val=&c3stat=&c3comp=&c3val=&c4stat=&c4comp=&c4val=&c5stat=&c5comp=&c6mult=&c6stat=&order_by=ws&order_by_asc=&offset=0"
url2 = "https://www.basketball-reference.com/play-index/psl_finder.cgi?request=1&match=combined&type=totals&per_minute_base=36&per_poss_base=100&lg_id=NBA&is_playoffs=N&year_min=1994&year_max=2020&franch_id=&season_start=1&season_end=-1&age_min=0&age_max=99&shoot_hand=&height_min=0&height_max=99&birth_country_is=Y&birth_country=&birth_state=&college_id=&draft_year=&is_active=&debut_yr_nba_start=&debut_yr_nba_end=&is_hof=&is_as=&as_comp=gt&as_val=1&award=&pos_is_g=Y&pos_is_gf=Y&pos_is_f=Y&pos_is_fg=Y&pos_is_fc=Y&pos_is_c=Y&pos_is_cf=Y&qual=&c1stat=&c1comp=&c1val=&c2stat=&c2comp=&c2val=&c3stat=&c3comp=&c3val=&c4stat=&c4comp=&c4val=&c5stat=&c5comp=&c6mult=&c6stat=&order_by=ws&order_by_asc=&offset=100"

nba_data = dataframe(url)
nba_data = nba_data.append(dataframe(url2))
nba_data.fillna(value=0,inplace=True) 
nba_data.reset_index(drop=True,inplace=True)
nba_data['Allstar'] = 1


#clean up dtypes       
for column in nba_data.columns:
    if column in ['Player','Tm','Lg']:
        nba_data[column] = nba_data[column].astype(str)
    else:
        nba_data[column] = nba_data[column].astype(float)

#create altered dataframe to join so the year they began in the NBA matches      
college_data['From'] = college_data['To'] + 1

#adjust for players who were injured in their first NBA year
college_data.loc[college_data['Player'] == 'Blake Griffin', 'From'] = 2011
college_data.loc[college_data['Player'] == 'Joel Embiid', 'From'] = 2017
college_data.loc[college_data['Player'] == 'Ben Simmons', 'From'] = 2018
 
#make a temp table joining the injury-adjusted and date-adjusted table
temp = pd.merge(college_data,nba_data[['Player','From','Allstar']],how='left',on=['Player','From'])
temp['Allstar'].fillna(value=0,inplace=True)
temp['Allstar'].sum()       
 
#find what NBA players are missing
missing = pd.merge(nba_data,temp,how='left',on=['Player','Allstar'])
missing['School'].fillna(value='X',inplace=True)
missing = missing.loc[missing['School'] == 'X']
missing_college = list(missing['Player'].unique())

#scrape for missing data 
missing = pd.DataFrame()
for player in missing_college:
    try:
        player2 = player.replace("'","")
        first_name = str.split(player2)[0].lower()
        last_name = str.split(player2)[-1].lower()
        url = "https://www.sports-reference.com/cbb/players/{0}-{1}-1.html".format(first_name,last_name)
        stats = requests.get(url)
        soup = BeautifulSoup(stats.content, 'html.parser')
        table = soup.findAll(name='table',attrs={'id':'players_per_game'})
        html_str = str(table)
        df = pd.read_html(html_str)[0]
        df['Rk'] = 0
        df['Player'] = player
        if int(df['Season'][0].split('-')[1]) > 21:
            From = int('19' + df['Season'][0].split('-')[1])
        else:
            From = int('20' + df['Season'][0].split('-')[1])
        df['From'] = From
        if int(df['Season'].head(-1).tail(1).values[0].split('-')[1]) > 21:
            To = int('19' + df['Season'].head(-1).tail(1).values[0].split('-')[1])
        else:
            To = int('20' + df['Season'].head(-1).tail(1).values[0].split('-')[1])
        df['To'] = To
        df['Conf'] = df['Conf'].head(1).values[0]
        if str(soup.findAll('p')[0].text).split()[0] == 'Position:':
            position = str(soup.findAll('p')[0].text).split()[1]
        else:
            position = str(soup.findAll('p')[1].text).split()[1]
        df['Position'] = position
        df = df.tail(1)
        missing = missing.append(df)
    except:
        print('No D1 college data found for',player,sort=True)
        pass        

#create proper columns
missing = missing[college_data.columns]


for column in missing.columns:
    if column in ['G','Rk','Player','From','To','School','Conf','Position']:
        continue
    else:
        missing[column] = missing[column] * missing['G']

#adjust position
def Position(x):
    if x == 'Center':
        return 'C'
    if x == 'Guard':
        return 'G'
    else:
        return 'F'

missing['Position'] = missing['Position'].apply(lambda x: Position(x))

#add to college_data

college_data = college_data.append(missing)    

#clean data types again
for column in college_data.columns:
    if column in ['Player','School','Conf','Position']:
        college_data[column] = college_data[column].astype(str)
    else:
        college_data[column] = college_data[column].astype(float)


#more stats just to have  
college_data['3P%'] = college_data['3P'] / college_data['3PA']
college_data['FG%'] = college_data['FG'] / college_data['FGA']
college_data['FT%'] = college_data['FT'] / college_data['FTA']
college_data['TS%'] = college_data['PTS'] / (2 * (college_data['FGA'] + (college_data['FTA'] * 0.44)))
college_data['A/TO'] = college_data['AST'] / college_data['TOV']
college_data['MPG'] =  college_data['MP'] / college_data['G']
college_data['Pts/36'] = college_data['PTS'] / college_data['MP'] * 36
college_data['Ast/36'] = college_data['AST'] / college_data['MP'] * 36
college_data['AST/G'] = college_data['AST'] / college_data['G']
college_data['Reb/36'] = college_data['TRB'] / college_data['MP'] * 36
college_data['Reb/G'] = college_data['TRB'] / college_data['G']
college_data['Pts/G'] = college_data['PTS'] / college_data['G']
college_data['Blk/G'] = college_data['BLK'] / college_data['G']
college_data['Stl/G'] = college_data['STL'] / college_data['G']

college_data.rename(columns={'To':'Season'},inplace=True)
college_data = pd.merge(college_data, nba_averages_by_year[['Season','TS%']], how='left',on='Season')
college_data.rename(columns={'TS%_x':'TS%','TS%_y':'lg_TS%','Season':'To'},inplace=True)

college_data['relTS'] = round((college_data['TS%'] / college_data['lg_TS%']) * 100)
college_data['PC/G'] = college_data['Pts/G'] + (college_data['AST/G'] * 2) + college_data['Reb/G'] + college_data['Reb/G'] + college_data['Stl/G'] - (college_data['PF'] / college_data['G']) - (college_data['TOV'] / college_data['G'])
college_data['rel PC/G'] = college_data['PC/G'] / college_data.groupby(['To','Position'])['PC/G'].transform('mean') * 100



temp = pd.merge(college_data,nba_data[['Player','Allstar']],how='left',on=['Player'])
#add 2020 allstars (actual selections to be made 1/23/20)
temp.loc[(temp['Player'] == 'Trae Young') | 
        (temp['Player'] == 'Pascal Siakam'), 'Allstar'] = 1
temp['Allstar'].fillna(value=0,inplace=True)
temp['Allstar'].sum()    

def Allstar_cleanup(row):
    if row['From'] > 2019:
        return 0
    elif (row['Player'] == 'Anthony Davis') & (row['School'] == 'Kentucky'):
        return 1
    elif row['Player'] == 'Anthony Davis':
        return 0
    elif (row['Player'] == 'Anthony Mason') & (row['School'] == 'Tennessee State'):
        return 1
    elif row['Player'] == 'Anthony Mason':
        return 0    
    elif (row['Player'] == 'Ben Simmons') & (row['School'] == 'Louisiana State'):
        return 1
    elif row['Player'] == 'Ben Simmons':
        return 0
    elif (row['Player'] == 'Danny Manning') & (['School'] == 'Kansas'):
        return 1
    elif row['Player'] == 'Danny Manning':
        return 0
    elif (row['Player'] == 'David Lee') & (row['School'] == 'Florida'):
        return 1
    elif row['Player'] == 'David Lee':
        return 0
    elif (row['Player'] == 'David Robinson') & (row['School'] == 'Navy'):
        return 1
    elif row['Player'] == 'David Robinson':
        return 0
    elif (row['Player'] == 'Derrick Coleman') & (row['School'] == 'Syracuse'):
        return 1
    elif row['Player'] == 'Derrick Coleman':
        return 0
    elif (row['Player'] == 'Devin Harris') & (row['School'] == 'Wisconsin'):
        return 1
    elif row['Player'] == 'Devin Harris':
        return 0
    elif (row['Player'] == 'Gary Payton') & (row['From'] == 1991):
        return 1
    elif row['Player'] == 'Gary Payton':
        return 0
    elif (row['Player'] == 'Glen Rice') & (row['School'] == 'Michigan'):
        return 1
    elif row['Player'] == 'Glen Rice':
        return 0
    elif (row['Player'] == 'Glenn Robinson') & (row['School'] == 'Purdue'):
        return 1
    elif row['Player'] == 'Glenn Robinson':
        return 0
    elif (row['Player'] == 'James Harden') & (row['School'] == 'Arizona'):
        return 1
    elif row['Player'] == 'James Harden':
        return 0
    elif (row['Player'] == 'Juwan Howard') & (row['School'] == 'Michigan'):
        return 1
    elif row['Player'] == 'Juwan Howard':
        return 0 
    elif (row['Player'] == 'Larry Johnson') & (row['School'] == 'UNLV'):
        return 1
    elif row['Player'] == 'Larry Johnson':
        return 0    
    elif (row['Player'] == 'Michael Jordan') & (row['School'] == 'UNC'):
        return 1
    elif row['Player'] == 'Michael Jordan':
        return 0    
    elif (row['Player'] == 'Mark Price') & (row['School'] == 'Georgia Tech'):
        return 1
    elif row['Player'] == 'Mark Price':
        return 0    
    elif (row['Player'] == 'Mo Williams') & (row['School'] == 'Alabama'):
        return 1
    elif row['Player'] == 'Mo Williams':
        return 0 
    elif (row['Player'] == 'Reggie Miller') & (row['School'] == 'UCLA'):
        return 1
    elif row['Player'] == 'Reggie Miller':
        return 0
    elif (row['Player'] == 'Sam Cassell') & (row['School'] == 'Florida State'):
        return 1
    elif row['Player'] == 'Sam Cassell':
        return 0    
    elif (row['Player'] == 'Steve Smith') & (row['School'] == 'Michigan State'):
        return 1
    elif row['Player'] == 'Steve Smith':
        return 0 
    elif (row['Player'] == 'Tim Hardaway') & (row['School'] == 'UTEP'):
        return 1
    elif row['Player'] == 'Tim Hardaway':
        return 0 
    elif row['Player'] == 'Tony Parker':
        return 0
    elif row['Player'] == 'Shawn Kemp':
        return 0
    else:
        return row['Allstar']
    
temp['Allstar'] = temp.apply(lambda row: Allstar_cleanup(row), axis = 1)




"""for another file"""

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import class_weight, resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier as GBC, RandomForestClassifier as RF
from sklearn.neural_network import MLPClassifier as MLP

temp.replace([np.inf, -np.inf], np.nan, inplace=True)
temp.fillna(value=0,inplace=True)

for column in temp.select_dtypes(include=['O']):
    temp[column], _ = pd.factorize(temp[column])

#set up x and y for modeling    
X = pd.DataFrame(temp.loc[temp['To'] < 2020].drop(['Rk','Player','From','To','Allstar'],axis=1))
y = temp.loc[temp['To'] < 2020, 'Allstar']   

Draft_2020 = pd.DataFrame(temp.loc[temp['To'] == 2020].drop(['Rk','Player','From','To','Allstar'],axis=1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


#need to do some sort of re-sampling as class is very imbalanced
#going to boost allstars to make a balance of 50/50 rather than the current .2/99.8

#from imblearn.over_sampling import SMOTE
#class_weights = {0:1,1:((len(y_train) - sum(y_train))/sum(y_train))}
X = pd.concat([X_train, y_train], axis=1)
not_allstar = X[X.Allstar==0]
allstar = X[X.Allstar==1]

allstar_upsampled = resample(allstar,
                          replace=True, # sample with replacement
                          n_samples=len(not_allstar), # match number in majority class for 50/50 class ratio
                          random_state=101) 

resampled = pd.concat([not_allstar, allstar_upsampled])

X_train = resampled.drop('Allstar',axis=1)
y_train = resampled.Allstar

#reset X for predictions later
X = pd.DataFrame(temp.loc[temp['To'] < 2020].drop(['Rk','Player','From','To','Allstar'],axis=1))

"""Gradient Boosting Machines"""


clf = GBC(n_estimators = 1000, random_state = 101, verbose = 3)
clf.fit(X_train, y_train)
clf.score(X_train, y_train)

y_pred = clf.predict_proba(X_test)[:,1]

GBM_auc = roc_auc_score(y_test,y_pred)

print("Gradient Boosting Machines")
print('GMB AUC:', GBM_auc)
print(confusion_matrix(y_test,clf.predict(X_test)))
print(classification_report(y_test,clf.predict(X_test)))

"""Random Forest"""

clf_RF = RF(n_estimators = 500, 
#            class_weight = class_weights,
            random_state = 101, 
            verbose = 3)

clf_RF.fit(X_train, y_train)
clf_RF.score(X_train, y_train)

RF_y_pred = clf_RF.predict_proba(X_test)[:,1]
RF_auc = roc_auc_score(y_test, RF_y_pred)

print("Random Forest")
print('Random Forest AUC:', RF_auc)
print(confusion_matrix(y_test,clf_RF.predict(X_test)))
print(classification_report(y_test,clf_RF.predict(X_test)))

"""Neural Network"""

param_grid = [{'activation':['logistic','tanh','relu'],
               'solver':['lbfgs','sgd','adam']}]
grid = GridSearchCV(MLP(random_state = 101, verbose = True),
                    param_grid,
                    cv=3, 
                    verbose=3)

grid.fit(X_train, y_train)
print(grid.best_score_)

NN_y_pred = grid.predict_proba(X_test)[:,1]
NN_auc = roc_auc_score(y_test, NN_y_pred)

print("Neural Network")
print('Neural Network AUC:', NN_auc)
print(confusion_matrix(y_test,grid.predict(X_test)))
print(classification_report(y_test,grid.predict(X_test)))

"""KNN"""

values_of_k = []
for i in range(1,round(math.sqrt(len(X_train) + 1))):
    if i % 2 == 0:
        continue
    else:
        values_of_k.append(i)

#take every 10th value, as testing too many values of k is extremely computationtally expensive
values_of_k = values_of_k[::10]

param_grid = {'n_neighbors':values_of_k}
grid_knn = GridSearchCV(KNeighborsClassifier(),param_grid,verbose=3, cv = 3)
grid_knn.fit(X_train, y_train)
grid_knn.score(X_train, y_train)

knn_y_pred = grid_knn.predict_proba(X_test)[:,1]
knn_auc = roc_auc_score(y_test,knn_y_pred)

print("KNeighbors")
print('KNeighbors AUC:', knn_auc)
print(confusion_matrix(y_test,grid_knn.predict(X_test)))
print(classification_report(y_test,grid_knn.predict(X_test)))

"""K-Means"""
from sklearn.cluster import KMeans

clf_KM = KMeans(n_clusters=2)
clf_KM.fit(X_train,y_train)

clf_KM.score(X_train, y_train)

KM_y_pred = clf_KM.predict(X_test)
KM_auc = roc_auc_score(y_test,KM_y_pred)

print("KMeansClustering")
print('KMeans AUC:', KM_auc)
print(confusion_matrix(y_test,clf_KM.predict(X_test)))
print(classification_report(y_test,clf_KM.predict(X_test)))


"""final predict"""
X = pd.DataFrame(temp.drop(['Rk','Player','From','To','Allstar'],axis=1))

college_data['Allstar Probability'] = clf_RF.predict_proba(X)[:,1]


#college_data.drop('Allstar prob',axis=1,inplace=True)

feature_importances = pd.DataFrame({'Feature': X. columns, 
                                    'Importance': clf_RF.feature_importances_.flatten()})
    
  
    
output = pd.DataFrame(college_data.loc[college_data['Allstar Probability'] > 0.001])
output = output[['Player', 'School','From', 'Conf','Position','Allstar Probability']]
output.sort_values(by=['From','Allstar Probability'],ascending=False,inplace=True)
output.rename(columns={'From':'Draft Year'},inplace=True)
output['Draft Year'] = output['Draft Year'] - 1



"""Visualization"""
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

#plot most important feature 
top_feature = feature_importances.loc[feature_importances['Importance'] == feature_importances['Importance'].max(), 'Feature']
college_data2 = college_data.loc[college_data['Allstar Probability'] >= .01]
plot_X = college_data2[top_feature].iloc[:,0]
plot_y = college_data2['Allstar Probability']

fig, ax = plt.subplots(figsize = (12,8))
ax = sns.scatterplot(plot_X,plot_y,alpha= 0.8,color='dodgerblue')
ax2 = sns.regplot(plot_X,plot_y,scatter=False, ci=None, color = 'dodgerblue')


#plot this feature for 2020 players who have at least a 1% chance of being an allstar
fig, ax = plt.subplots(figsize = (12,8))
players2020 = college_data.loc[(college_data['To'] == 2020) & 
                               college_data['Allstar Probability'] >= .01]
plot_X = players2020[top_feature].iloc[:,0]
plot_y = players2020['Allstar Probability']
ax = sns.scatterplot(plot_X,plot_y,alpha= 0.8,color='dodgerblue')

top10_upcoming = players2020.sort_values(by='Allstar Probability',ascending=False).head(10)
top10_upcoming.reset_index(drop=True,inplace=True)
ax2 = sns.scatterplot(top10_upcoming[top_feature].iloc[:,0], top10_upcoming['Allstar Probability'],alpha= 0.8,color='red')

for i in range(len(top10_upcoming)):
    player = top10_upcoming['Player'][i]
    x = top10_upcoming[top_feature].iloc[:,0][i]
    y = top10_upcoming['Allstar Probability'][i]
    if player == 'Cole Anthony':
        plt.annotate(player, xy=(x,y),horizontalalignment='left',verticalalignment='bottom')
    elif player == 'James Wiseman':
        plt.annotate(player, xy=(x,y),horizontalalignment='left',verticalalignment='top')
    else:
        plt.annotate(player, xy=(x,y),horizontalalignment='right',verticalalignment='bottom')

plt.text(x = -((ax.get_xticks()[1] - ax.get_xticks()[0]) / 2), y = ax.get_ylim()[1] * 1.1,
         fontsize = 26, weight = 'bold', alpha = 0.75,
         s = 'NBA All-Star probabilities for current college players')
plt.text(x = -((ax.get_xticks()[1] - ax.get_xticks()[0]) / 2), y = ax.get_ylim()[1] * 1.05,
         fontsize = 20, alpha = 0.85,
         s = 'Plotted agains most important feature, top 10 players highlighted')