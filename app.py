'''
The following code has certain constraints and based on assumptions.

'''


#importing packages used in the code.
import pandas as pd
import csv
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


#creating dataframes for the given datasets
df_players = pd.read_csv("Players.csv", index_col=0)
df_seasons = pd.read_csv("Seasons_Stats.csv",index_col=0)
df_players_data= pd.read_csv("player_data.csv", index_col=0)
#limiting the dataset of seasons
df_limited =df_seasons.loc[:, ["Player", "G", "PTS", "Age", "Year", "Pos"]]

#creating the dataframe to get player names in sorted order from players.csv
df_names = df_players.sort_values(by="Player")
df_names.index= np.arange(0,3922)
names = df_names["Player"]

'''attaining the maximum age, games played , posiiton played at
and points made for each player'''
max_age = df_limited.groupby("Player")["Age"].max()
max_games= df_limited.groupby("Player")["G"].max()
max_points = df_limited.groupby("Player")["PTS"].max()
max_pos = df_limited.groupby("Player")["Pos"].max()

#creating a list to accumulate data in consolidated manner
players_list =[]

for i in range(0,3921):
    name = names[i]
    age = max_age[i]
    games = max_games[i]
    points = max_points[i]
    position = max_pos[i]

    players_list.append([name,age, games, points, position ])


#converting the list into a dataframe
pl = pd.DataFrame(players_list)

#replacing NaN values in age wiht 0
pl[1] = pl[1].replace(np.nan, 0)

#using age, games played and points as variables to train model
X = pl.iloc[:, [1,2,3]]

'''after using the elbow chart, 2 clusters are relevant
    using K-Means clustering to group data
'''
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

#adding the cluster info in the dataframe
pl[5] = y_kmeans

#creating separate dataframes for clusters
pl1 = pl[pl[5]==0]
pl2 = pl[pl[5]==1]

#plotting the graph to visualize the clusters
kplot = plt.axes(projection='3d')
xline = np.linspace(0, 15, 1000)
yline = np.linspace(0, 15, 1000)
zline = np.linspace(0, 15, 1000)
kplot.plot3D(xline, yline, zline, 'black')
# Data for three-dimensional scattered points
kplot.scatter3D(pl1[1], pl1[2], pl1[3], c='red', label = 'Cluster 1')
kplot.scatter3D(pl2[1],pl2[2],pl2[3],c ='green', label = 'Cluster 2')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color = 'indigo', s = 200)
plt.legend()
plt.title("Kmeans")
#plt.show()


#sorting the data in cluster 1 on age and games played in ascending order
best_age = pl1.sort_values(by=[1,2], ignore_index=True)
best_age_list=best_age.values.tolist()

#soritng the data in cluster 1 on points in descending order
best_points = pl1.sort_values(by=[3], ignore_index=True, ascending=False)
best_points_list=best_points.values.tolist()

#taking input from user for specific points
sum = float(input("Input a specific points sum for a team:\n"))

sum_needed =0
selected_player = []



#selecting players for given sum based on age and games
for i in range(0,1236):
    while sum_needed < sum:
        if len(selected_player) < 5:
            if best_age_list[i][3] > float(sum-sum_needed):
                if len(selected_player) == 4:
                    sum_needed += best_age_list[i][3]
                    selected_player.append(best_age_list[i])
                else:
                    pass
            else:
                sum_needed += best_age_list[i][3]
                selected_player.append(best_age_list[i])
                break
        else:
            selected_player.pop(-1)

        break

#selecting players based on points
for i in range(0,1236):
    while sum_needed < sum:
        if len(selected_player) < 5:
            if best_points_list[i][3] > float(sum-sum_needed):
                if len(selected_player) == 4:
                    sum_needed += best_points_list[i][3]
                    selected_player.append(best_points_list[i])
                else:
                    pass
            else:
                sum_needed += best_points_list[i][3]
                selected_player.append(best_points_list[i])
                break
        else:
            selected_player.pop(-1)

        break



print("The suggested team of five players:\n")

for plr in selected_player:
    print(f"Player {plr[0]} with age {plr[1]}, games played {plr[2]} and points {plr[3]}")
    print('-'*20)




