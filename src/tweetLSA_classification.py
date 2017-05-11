#!/usr/bin/env python
"""
Run k-NN classification on the Reuters text dataset using LSA.
This script leverages modules in scikit-learn for performing tf-idf and SVD.
Classification is performed using k-NN with k=5 (majority wins).
The script measures the accuracy of plain tf-idf as a baseline, then LSA to
show the improvement.
@author: Chris McCormick
"""

import pickle
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier

save = 1
###############################################################################
#  Load the raw text dataset.
###############################################################################

print("Loading dataset...")

# The raw text dataset is stored as tuple in the form:
# (X_train_raw, y_train_raw, X_test_raw, y_test)
# The 'filtered' dataset excludes any articles that we failed to retrieve
# fingerprints for.

text_file = open("/Users/amritaanam/PycharmProjects/FlashFloodTwitter/classification/X_train_raw.txt")
X_train_raw = text_file.read().split('\n')
text_file = open("/Users/amritaanam/PycharmProjects/FlashFloodTwitter/classification/Y_train_labels.txt")
y_train = text_file.read().split('\n')
text_file = open("/Users/amritaanam/PycharmProjects/FlashFloodTwitter/classification/X_test_raw.txt")
X_test_raw = text_file.read().split('\n')
text_file = open("/Users/amritaanam/PycharmProjects/FlashFloodTwitter/classification/Y_test_labels.txt")
y_test = text_file.read().split('\n')

print((len(X_train_raw), len(y_train)))
print((len(X_test_raw), len(y_test)))


###############################################################################
#  Use LSA to vectorize the articles.
###############################################################################

# Tfidf vectorizer:
#    Strips out stop words
#   - Filters out terms that occur in more than half of the docs (max_df=0.5)
#   - Filters out terms that occur in only one document (min_df=2).
#   - Selects the 10,000 most frequently occuring words in the corpus.
#   - Normalizes the vector (L2 norm of 1.0) to normalize the effect of
#     document length on the tf-idf values.
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                             min_df=2, stop_words='english',
                             use_idf=True)

# Build the tfidf vectorizer from the training data ("fit"), and apply it
# ("transform").
X_train_tfidf = vectorizer.fit_transform(X_train_raw)

print("  Actual number of tfidf features: %d" % X_train_tfidf.get_shape()[1])

print("\nPerforming dimensionality reduction using LSA")
t0 = time.time()

# Project the tfidf vectors onto the first 150 principal components.
# Though this is significantly fewer features than the original tfidf vector,
# they are stronger features, and the accuracy is higher.
svd = TruncatedSVD(100)
lsa = make_pipeline(svd, Normalizer(copy=False))

# Run SVD on the training data, then project the training data.
X_train_lsa = lsa.fit_transform(X_train_tfidf)

print("  done in %.3fsec" % (time.time() - t0))

explained_variance = svd.explained_variance_ratio_.sum()
print("  Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))


# Now apply the transformations to the test data as well.
X_test_tfidf = vectorizer.transform(X_test_raw)
X_test_lsa = lsa.transform(X_test_tfidf)


###############################################################################
#  Run classification of the test articles
###############################################################################

print("\nClassifying tfidf vectors...")

# Time this step.
t0 = time.time()

# Build a k-NN classifier. Use k = 5 (majority wins), the cosine distance,
# and brute-force calculation of distances.
knn_tfidf = KNeighborsClassifier(n_neighbors=3, algorithm='brute', metric='cosine')
knn_tfidf.fit(X_train_tfidf, y_train)

# Classify the test vectors.
p = knn_tfidf.predict(X_test_tfidf)

# Measure accuracy
numRight = 0;
for i in range(0,len(p)):
    if p[i] == y_test[i]:
        numRight += 1

print("  (%d / %d) correct - %.2f%%" % (numRight, len(y_test), float(numRight) / float(len(y_test)) * 100.0))

# Calculate the elapsed time (in seconds)
elapsed = (time.time() - t0)
print("  done in %.3fsec" % elapsed)


print("\nClassifying LSA vectors...")

# Time this step.
t0 = time.time()

# Build a k-NN classifier. Use k = 5 (majority wins), the cosine distance,
# and brute-force calculation of distances.
knn_lsa = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='cosine')
knn_lsa.fit(X_train_lsa, y_train)

# Classify the test vectors.
p = knn_lsa.predict(X_test_lsa)
#print (p)


if save == 1:
    np.savetxt("/Users/amritaanam/PycharmProjects/FlashFloodTwitter/classification/predict_test.csv", p, delimiter=",", fmt="%s")
# Measure accuracy
numRight = 0;
for i in range(0,len(p)):
    if p[i] == y_test[i]:
        numRight += 1

print("  (%d / %d) correct - %.2f%%" % (numRight, len(y_test), float(numRight) / float(len(y_test)) * 100.0))

# Calculate the elapsed time (in seconds)
elapsed = (time.time() - t0)
print("    done in %.3fsec" % elapsed)
