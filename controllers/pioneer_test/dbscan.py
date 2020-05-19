import pandas as pd
from sklearn.cluster import DBSCAN
import plotly.express as px

rangeImageCompleteDf = pd.read_csv('rangeImageComplete.csv')
dbscan = DBSCAN(eps=11).fit(rangeImageCompleteDf)
rangeImageCompleteDf['label'] = dbscan.labels_
fig = px.scatter(rangeImageCompleteDf, y='label')
fig.show()
