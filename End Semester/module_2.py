import time
import redis
from flask import current_app
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


import pandas as pd


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




def info(msg):
    current_app.logger.info(msg)


class ContentEngine(object):

    SIMKEY = 'p:smlr:%s'

    def __init__(self):
        self._r = redis.StrictRedis.from_url(current_app.config['REDIS_URL'])

    def train(self, data_source):
        start = time.time()
        ds = pd.read_csv(data_source)
        info("Training data ingested in %s seconds." % (time.time() - start))

        # Flush the stale training data from redis
        self._r.flushdb()

        start = time.time()
        self._train(ds)
        info("Engine trained in %s seconds." % (time.time() - start))

    def _train(self, ds):
        """
        Train the engine.
        Create a TF-IDF matrix of unigrams, bigrams, and trigrams for each product. The 'stop_words' param
        tells the TF-IDF module to ignore common english words like 'the', etc.
        Then we compute similarity between all products using SciKit Leanr's linear_kernel (which in this case is
        equivalent to cosine similarity).
        Iterate through each item's similar items and store the 100 most-similar. Stops at 100 because well...
        how many similar products do you really need to show?
        Similarities and their scores are stored in redis as a Sorted Set, with one set for each item.
        :param ds: A pandas dataset containing two fields: description & id
        :return: Nothin!
        """
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
        tfidf_matrix = tf.fit_transform(ds['description'])

        cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

        for idx, row in ds.iterrows():
            similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
            similar_items = [(cosine_similarities[idx][i], ds['id'][i]) for i in similar_indices]

            # First item is the item itself, so remove it.
            # This 'sum' is turns a list of tuples into a single tuple: [(1,2), (3,4)] -> (1,2,3,4)
            flattened = sum(similar_items[1:], ())
            self._r.zadd(self.SIMKEY % row['id'], *flattened)

    def predict(self, item_id, num):
        """
        Couldn't be simpler! Just retrieves the similar items and their 'score' from redis.
        :param item_id: string
        :param num: number of similar items to return
        :return: A list of lists like: [["19", 0.2203], ["494", 0.1693], ...]. The first item in each sub-list is
        the item ID and the second is the similarity score. Sorted by similarity score, descending.
        """
        return self._r.zrange(self.SIMKEY % item_id, 0, num-1, withscores=True, desc=True)


content_engine = ContentEngine()