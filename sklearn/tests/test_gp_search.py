"""
Testing for grid search module (sklearn.grid_search)

"""

from sklearn.externals.six.moves import cStringIO as StringIO
import sys

import numpy as np

from sklearn.utils.testing import assert_equal, assert_raises,\
    ignore_warnings, assert_raise_message
from sklearn.svm import LinearSVC
from sklearn.utils.testing import assert_true
from sklearn.datasets import make_blobs
from sklearn.gp_search import GPSearchCV


# Neither of the following two estimators inherit from BaseEstimator,
# to test hyperparameter search on user-defined classifiers.
class MockClassifier(object):
    """Dummy classifier to test the cross-validation"""
    def __init__(self, foo_param=0):
        self.foo_param = foo_param

    def fit(self, X, Y):
        assert_true(len(X) == len(Y))
        return self

    def predict(self, T):
        return T.shape[0]

    predict_proba = predict
    decision_function = predict
    transform = predict

    def score(self, X=None, Y=None):
        if self.foo_param == 2:
            score = 1.
        else:
            score = 0.
        return score

    def get_params(self, deep=False):
        return {'foo_param': self.foo_param}

    def set_params(self, **params):
        self.foo_param = params['foo_param']
        return self


class LinearSVCNoScore(LinearSVC):
    """An LinearSVC classifier that has no score method."""
    @property
    def score(self):
        raise AttributeError

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])


def test_gp_search():
    # Test that the best estimator contains the right value for foo_param
    clf = MockClassifier()
    gp_search = GPSearchCV(clf, {'foo_param': [1, 2, 3]}, verbose=3)
    # make sure it selects the smallest parameter in case of ties
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    gp_search.fit(X, y)
    sys.stdout = old_stdout
    assert_equal(gp_search.best_estimator_.foo_param, 2)

    # Smoke test the score etc:
    gp_search.score(X, y)
    gp_search.predict_proba(X)
    gp_search.decision_function(X)
    gp_search.transform(X)

    # Test exception handling on scoring
    gp_search.scoring = 'sklearn'
    assert_raises(ValueError, gp_search.fit, X, y)


# @ignore_warnings
# def test_grid_search_no_score():
#     # Test grid-search on classifier that has no score function.
#     clf = LinearSVC(random_state=0)
#     X, y = make_blobs(random_state=0, centers=2)
#     Cs = [.1, 1, 10]
#     clf_no_score = LinearSVCNoScore(random_state=0)
#     gp_search = GPSearchCV(clf, {'C': Cs}, scoring='accuracy')
#     gp_search.fit(X, y)

#     grid_search_no_score = GPSearchCV(clf_no_score, {'C': Cs},
#                                         scoring='accuracy')
#     # smoketest grid search
#     grid_search_no_score.fit(X, y)

#     # check that best params are equal
#     assert_equal(grid_search_no_score.best_params_, gp_search.best_params_)
#     # check that we can call score and that it gives the correct result
#     assert_equal(gp_search.score(X, y), grid_search_no_score.score(X, y))

#     # giving no scoring function raises an error
#     grid_search_no_score = GPSearchCV(clf_no_score, {'C': Cs})
#     assert_raise_message(TypeError, "no scoring", grid_search_no_score.fit,
#                          [[1]])

