#!/usr/bin/python

import numpy as np
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as pl
import time
import warnings

warnings.filterwarnings("ignore")


# performs Latent Semantic Analysis on tweets
def LSA(stopset, tweets):
    # Vectorize the tweets with tf-idf weights where each row is a tweet and each column is a word
    vectorizer = TfidfVectorizer(stop_words=stopset, use_idf=True, ngram_range=(1, 1))
    X = vectorizer.fit_transform(tweets)
    lsa = TruncatedSVD(n_components=200, n_iter=(10000)).fit(X)
    X_lsa = lsa.fit_transform(X)

    return lsa, X_lsa

# computes distance (1-cosine similarity) between documents
# input V matrix from LSA (rows = number of documents, columns = singular vectors)
def docDistance(D):
    sim_mat = np.zeros((D.shape[0], D.shape[0]))
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            doc_sim = 1 - cosine_similarity(D[i], D[j])
            sim_mat[i][j] = doc_sim
            j = j + 1
        i = i + 1

    return (sim_mat)


def conceptTerms(terms, lsa):
    for i, comp in enumerate(lsa.components_):
        termsInComp = zip(terms, comp)
        sortedTerms = sorted(termsInComp, key=lambda x: x[1], reverse=True)[:5]
        print("Concept %d:" % i)
        for term in sortedTerms:
            print(term[0])
        print(" ")


def KMeansClustering(X):
    kmeans = KMeans(n_clusters=3, random_state=1, max_iter=10000).fit(X)

    pca = PCA(copy=True, n_components=2, random_state=1).fit(X)
    pca_2d = pca.transform(X)

    cluster_data = np.zeros((X.shape[0], 3))
    for i in range(0, pca_2d.shape[0]):
        cluster_data[i][0] = kmeans.labels_[i]
        cluster_data[i][1] = pca_2d[i][0]
        cluster_data[i][2] = pca_2d[i][1]


    return kmeans, cluster_data, pca, pca_2d


def plotCluster(kmeans, pca_2d):
    # pl.ion()
    for i in range(0, pca_2d.shape[0]):
        if kmeans.labels_[i] == 0:
            c1 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='*')
        elif kmeans.labels_[i] == 1:
            c2 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='+')
        elif kmeans.labels_[i] == 2:
            c3 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='o')
       # elif kmeans.labels_[i] == 3:
       #     c4 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='y', marker='x')
    pl.legend([c1, c2, c3], ['c0', 'c1', 'c2'])
    pl.title('Ellicott city flash flood tweets ')
    pl.show()


def main(in_file, out_folder, save):
    # open input file with all tweets separated by a new_line
    text_file = open(in_file)

    # read all the tweets, eliminate numbers and white sapces and store them in a list where each element is a tweet
    tweets = (text_file.read().strip()).translate(None, '1234567890').split('\n')

    # define a dictionary of stop words
    stopset = set(stopwords.words('english'))
    freq_words = ["ellicott", "city", "ellicottcity", "md", "maryland", "in", "elliott", "#"]
    stopset |= set(freq_words)

    # apply lsa to fit the vectorized tweet matrix (X), and save the result
    print("LSA:")

    start_time_lsa = time.time()
    lsa, X_lsa = LSA(stopset, tweets)
    time_taken_lsa = time.time() - start_time_lsa

    print("LSA variance:" + str("%.2f" % (sum(lsa.explained_variance_) * 100)))
    print("Time: " + str("%.2f" % time_taken_lsa) + "sec")
    print("*****************************")
    print("\n")

    if save == 1:
        np.savetxt(out_folder + "X_lsa.csv", X_lsa, delimiter=",")
        np.savetxt(out_folder + "X_lsa_variance.csv", lsa.explained_variance_, delimiter=",")

    # compute tweet-tweet distance matrix
    print("Tweet Distance:")

    start_time_dis = time.time()
    S = docDistance(X_lsa)
    time_taken_dis = time.time() - start_time_dis

    print("Time: " + str("%.2f" % time_taken_dis) + "sec")
    print("*****************************")
    print("\n")

    if save == 1:
        np.savetxt(out_folder + "tweet_distance.csv", S, delimiter=",")

    # compute k-means clustering on tweet distance matrix
    # save the cluster indexes with the top 2 principal components and the cluster centroids
    print("k-means:")

    start_time_km = time.time()
    K, C, P, P2D = KMeansClustering(S)
    time_taken_km = time.time() - start_time_km

    print("PCA variance:" + str("%.2f" % (sum(P.explained_variance_) * 100)))
    print("Time: " + str("%.2f" % time_taken_km) + "sec")
    print("*****************************")
    print("\n")

    if save == 1:
        np.savetxt(out_folder + "cluster_centroids.csv", K.cluster_centers_, delimiter=",")
        np.savetxt(out_folder + "tweet_clusters.csv", C, delimiter=",")

    # plot the clusters
    print("Plot:")
    plotCluster(K, P2D)

    # concept terms
    # terms = vectorizer.get_feature_names()
    # conceptTerms(terms, lsa)


in_file = "/Users/amritaanam/PycharmProjects/FlashFloodTwitter/train/ec_ff_tweets.txt"
out_folder = "/Users/amritaanam/PycharmProjects/FlashFloodTwitter/train/"
main(in_file, out_folder, 1)
