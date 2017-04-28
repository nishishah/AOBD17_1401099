# --- Import Libraries --- #

import pandas as pd
from scipy.spatial.distance import cosine


from sklearn.cluster import KMeans
from numpy import *

from numpy.random import *

import matplotlib.pyplot as plt
import matplotlib as mpl

data = pd.read_csv('data.csv')

X = randint(300,size=(100,2))


plt.ion()
initlabel = None
for i in range(1,1000):
    y = randint(300,size=(1,2))
    X = concatenate((X,y),axis=0)
    kmeans = KMeans(n_clusters=4).fit(X)
    labels = array(kmeans.labels_)
    print labels
    if initlabel is None:
        initlabel = labels[:]
    else:
        dict1 = {}
        for i in range(20):
            dict1[labels[i]] = initlabel[i]
        labels[:initlabel.shape[0]] = initlabel
        labels[-1] = dict1[labels[-1]]
        initlabel = labels[:]
    means = array(kmeans.cluster_centers_)
    """
    cl1 = labels == 0
    cl2 = labels == 1
    cl3 = labels == 2
    cl4 = labels == 3
    cl1 = X[cl1]
    cl2 = X[cl2]
    cl3 = X[cl3]
    cl4 = X[cl4]
    """
    colors = ['red','yellow','blue','magenta']
    plt.clf()
    plt.scatter(X[:,0],X[:,1],c=labels,cmap=mpl.colors.ListedColormap(colors))
    plt.plot(means[:,0],means[:,1],'ks',ms=7)
    plt.pause(0.05)




# --- Read Data --- #


# --- Start Item Based Recommendations --- #
# Drop any column named "user"
data_germany = data.drop('user', 1)

# Create a placeholder dataframe listing item vs. item
data_ibs = pd.DataFrame(index=data_germany.columns, columns=data_germany.columns)

# Lets fill in those empty spaces with cosine similarities
# Loop through the columns
for i in range(0, len(data_ibs.columns)):
    # Loop through the columns for each column
    for j in range(0, len(data_ibs.columns)):
        # Fill in placeholder with cosine similarities
        data_ibs.ix[i, j] = 1 - cosine(data_germany.ix[:, i], data_germany.ix[:, j])

# Create a placeholder items for closes neighbours to an item
data_neighbours = pd.DataFrame(index=data_ibs.columns, columns=[range(1, 11)])

# Loop through our similarity dataframe and fill in neighbouring item names
for i in range(0, len(data_ibs.columns)):
    data_neighbours.ix[i, :10] = data_ibs.ix[0:, i].order(ascending=False)[:10].index


# --- End Item Based Recommendations --- #

# --- Start User Based Recommendations --- #

# Helper function to get similarity scores
def getScore(history, similarities):
    return sum(history * similarities) / sum(similarities)


# Create a place holder matrix for similarities, and fill in the user name column
data_sims = pd.DataFrame(index=data.index, columns=data.columns)
data_sims.ix[:, :1] = data.ix[:, :1]

# Loop through all rows, skip the user column, and fill with similarity scores
for i in range(0, len(data_sims.index)):
    for j in range(1, len(data_sims.columns)):
        user = data_sims.index[i]
        product = data_sims.columns[j]

        if data.ix[i][j] == 1:
            data_sims.ix[i][j] = 0
        else:
            product_top_names = data_neighbours.ix[product][1:10]
            product_top_sims = data_ibs.ix[product].order(ascending=False)[1:10]
            user_purchases = data_germany.ix[user, product_top_names]

            data_sims.ix[i][j] = getScore(user_purchases, product_top_sims)

# Get the top songs
data_recommend = pd.DataFrame(index=data_sims.index, columns=['user', '1', '2', '3', '4', '5', '6'])
data_recommend.ix[0:, 0] = data_sims.ix[:, 0]

# Instead of top song scores, we want to see names
for i in range(0, len(data_sims.index)):
    data_recommend.ix[i, 1:] = data_sims.ix[i, :].order(ascending=False).ix[1:7, ].index.transpose()

# Print a sample
print data_recommend.ix[:10, :4]