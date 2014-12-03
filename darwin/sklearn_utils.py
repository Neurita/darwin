# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
#Authors:
# Alexandre Manhaes Savio <alexsavio@gmail.com>
# Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
# Neurita S.L.
#
# BSD 3-Clause License
#
# 2014, Alexandre Manhaes Savio
# Use this at your own risk!
#------------------------------------------------------------------------------

import logging

#cross-validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut
from sklearn.cross_validation import StratifiedKFold

#pipelining
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union


#instances
from darwin.instance import SelectorInstantiator, LearnerInstantiator

log = logging.getLogger(__name__)


def get_clfmethod(clfmethod):
    """Return a classification method and a classifiers parameter grid-search

    Parameters
    ----------
    clfmethod: str
        clfmethod posible choices: 'DecisionTreeClassifier', 'RBFSVC', 'PolySVC',
                                    'LinearSVC', 'GMM', 'RandomForestClassifier',
                                    'ExtraTreesClassifier', SGDClassifier',
                                    'Perceptron'

    Returns
    -------
    classifier, param_grid
    """

    learner_instance = LearnerInstantiator()
    try:
        return learner_instance.get_method_with_grid(clfmethod)
    except:
        log.exception("Error: {} should be in {}".format(clfmethod, learner_instance.methods))
        raise


def get_fsmethod(fsmethod):
    """Creates a feature selection method and a parameter grid-search.

    Parameters
    ----------
    fsmethod: string
        fsmethod choices: 'rfe', 'rfecv', 'selectPercentile', 'selectFpr', 'SelectFdr',
                      'ExtraTreesClassifier', 'PCA', 'RandomizedPCA', 'LDA', 'SelectKBest',
                      'PearsonCorrelationSelection', 'BhatacharyyaGaussianSelection',
                       'WelchTestSelection'

    Returns
    -------
    fsmethods[fsmethod], fsgrid[fsmethod]
    """

    selector_instance = SelectorInstantiator()
    try:
        return selector_instance.get_method_with_grid(fsmethod)
    except:
        log.exception("Error: {} should be in {}".format(fsmethod, selector_instance.methods))
        raise


def get_cv_method(targets, cvmethod='10', stratified=True):
    """Creates a cross-validation object

    Parameters
    ----------
    targets   : list or vector
        Class labels set in the same order as in X

    cvmethod  : string or int
        String of a number or number for a K-fold method, 'loo' for LeaveOneOut

    stratified: bool
        Indicates whether to use a Stratified K-fold approach

    Returns
    -------
    Returns a class from sklearn.cross_validation
    """
    n = len(targets)

    if cvmethod == 'loo':
        return LeaveOneOut(n)

    if stratified:
        if isinstance(cvmethod, int):
            return StratifiedKFold(targets, cvmethod)
        elif isinstance(cvmethod, str):
            if cvmethod.isdigit():
                return StratifiedKFold(targets, int(cvmethod))
    else:
        if isinstance(cvmethod, int):
            return KFold(n, cvmethod)

        elif isinstance(cvmethod, str):
            if cvmethod.isdigit():
                return KFold(n, int(cvmethod))

    return StratifiedKFold(targets, int(cvmethod))


def get_pipeline(fsmethods, clfmethod):
    """Returns an instance of a sklearn Pipeline given the parameters
    fsmethod1 and fsmethod2 will be joined in a FeatureUnion, then it will joined
    in a Pipeline with clfmethod

    Parameters
    ----------
    fsmethods: list of estimators
        All estimators in a pipeline, must be transformers (i.e. must have a transform method).

    clfmethod: classifier
        The last estimator may be any type (transformer, classifier, etc.).

    Returns
    -------
    pipe
    """
    feat_union = None
    if not isinstance(fsmethods, list):
        if hasattr(fsmethods, 'transform'):
            feat_union = fsmethods
        else:
            raise ValueError('fsmethods expected to be either a list or a transformer method')
    else:
        feat_union = make_union(*fsmethods)

    if feat_union is None:
        pipe = make_pipeline(clfmethod)
    else:
        pipe = make_pipeline(feat_union, clfmethod)

    return pipe
