from sklearn.datasets import load_digits
from sklearn.gp_search import GPSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

import logging
import matplotlib.pyplot as plt
import numpy as np

"""
Compare GP-based search vs random search on iris and text datasets.
"""

test_name = 'text'
test_name = 'iris'
n_tests = 20
# n_tests = 10
n_iter_search = 60
# n_iter_search = 10
n_init = 20
save_data=True

print('Test GP vs Random on ', test_name, 'dataset - Average on ', n_tests,' trials')


def extend_result(n_tests, tmp_res):
    res = np.zeros(n_tests)
    l = len(tmp_res) - 1
    for i in range(n_tests):
        res[i] = tmp_res[min(i, l)]
    return res



if test_name == 'iris':
    iris = load_digits()
    X, y = iris.data, iris.target
    pipeline = RandomForestClassifier()

    # specify parameters and distributions to sample from
    parameters = {"max_depth": ['int', [3, 3]],
                  "max_features": ['int', [1, 11]],
                  "min_samples_split": ['int', [1, 11]],
                  "min_samples_leaf": ['int', [1, 11]],
                  "bootstrap": ['cat', [True, False]],
                  "criterion": ['cat', ["gini", "entropy"]]}

elif test_name == 'text':
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # Load some categories from the training set
    categories = [
        'alt.atheism',
        'talk.religion.misc',
    ]
    # Uncomment the following to do the analysis on all the categories
    # categories = None
    print("Loading 20 newsgroups dataset for categories:")
    print(categories)

    data = fetch_20newsgroups(subset='train', categories=categories)
    print("%d documents" % len(data.filenames))
    print("%d categories" % len(data.target_names))

    X = data.data
    y = data.target

    # define a pipeline combining a text feature extractor with a simple
    # classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),
    ])

    # uncommenting more parameters will give better exploring power but will
    # increase processing time in a combinatorial way
    parameters = {
        'vect__max_df': ['float', [0.5, 1.]],
        # 'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ['cat', [(1, 1), (1, 2)]],  # unigrams or bigrams
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': ['float', [0.000001, 0.00001]],
        'clf__penalty': ['cat', ['l2', 'elasticnet']]
        # 'clf__n_iter': (10, 50, 80),
    }

else:
    print('Dataset not available for test')

# GP UCB search
all_gp_ucb_results = []
print('GP_ucb search')
for i in range(n_tests):
    ucb_search = GPSearchCV(pipeline, parameters,
                            acquisition_function='UCB',
                            n_iter=n_iter_search, 
                            n_init=n_init, 
                            verbose=False).fit(X, y)

    scores = ucb_search.scores_  
    max_scores = [scores[0]]
    print('Test', i, '-', len(scores), 'parameters tested')

    for j in range(1, len(scores)):
        max_scores.append(max(max_scores[j-1], scores[j]))
    all_gp_ucb_results.append(extend_result(n_iter_search, max_scores))

all_gp_ucb_results = np.asarray(all_gp_ucb_results)
print(all_gp_ucb_results.shape)
if save_data:
    np.savetxt('gp_ucb_scores.csv', all_gp_ucb_results, delimiter=',')

# GP EI search
all_gp_ei_results = []
print('GP_ei search')

for i in range(n_tests):
    ei_search = GPSearchCV(pipeline, parameters,
                           acquisition_function='EI',
                           n_iter=n_iter_search, 
                           n_init=20, 
                           verbose=False).fit(X, y)
  
    scores = ei_search.scores_

    max_scores = [scores[0]]
    print('Test',i,'-',len(scores),'parameters tested')

    for j in range(1,len(scores)):
        max_scores.append(max(max_scores[j-1],scores[j]))
    all_gp_ei_results.append(extend_result(n_iter_search,max_scores))

all_gp_ei_results = np.asarray(all_gp_ei_results)
print(all_gp_ei_results.shape)


# Randomized search
print('Random search')
all_random_results = []
for i in range(n_tests):
    random_search = RandomizedSearchCV(
        pipeline, parameters, n_iter=n_iter_search,
        n_init=n_iter_search, verbose=False).fit(X, y)

    scores = random_search.scores_

    max_scores = [scores[0]]
    print('Test', i, '-', len(scores), 'parameters tested')

    for j in range(1, len(scores)):
        max_scores.append(max(max_scores[j-1], scores[j]))
    all_random_results.append(extend_result(n_iter_search, max_scores))
all_random_results = np.asarray(all_random_results)

if save_data:
    np.savetxt('rand_scores.csv', all_random_results, delimiter=',')

plt.figure()
plt.plot(range(n_iter_search),np.mean(all_gp_ei_results,axis=0),'r',label='GP-EI')
plt.plot(range(n_iter_search), np.mean(all_gp_ucb_results, axis=0), 'b', label='GP-UCB')
plt.plot(range(n_iter_search), np.mean(all_random_results, axis=0), 'g', label='Random')
plt.legend(loc=4)
plt.title('Test GP vs Random on ' + test_name + ' dataset - Average on ' + str(n_tests) + ' trials')
plt.xlabel('Iterations')
plt.ylabel('Max CV performance')
plt.show()

