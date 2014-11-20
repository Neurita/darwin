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

import numpy as np
import logging

#classification
from sklearn import tree
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.mixture import GMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron

#feature selection
from sklearn.ensemble import ExtraTreesClassifier

#cross-validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut
from sklearn.cross_validation import StratifiedKFold

#pipelining
from sklearn.pipeline import Pipeline, FeatureUnion
from .utils.strings import append_to_keys

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


def get_pipeline(fsmethod1, fsmethod2, clfmethod, n_feats, n_cpus,
                 fs1_kwargs={}, fs2_kwargs={}, clf_kwargs={}):
    """Returns an instance of a sklearn Pipeline given the parameters

    Parameters
    ----------
    fsmethod1: str
        See get_fsmethod docstring for valid values

    fsmethod2: str
        See get_fsmethod docstring for valid values

    clfmethod: str
        See get_clfmethod docstring for valid values

    n_feats: int
        Number of features

    n_cpus: int

    fs1_kwargs: dict

    fs2_kwargs: dict

    clf_kwargs: dict

    Returns
    -------
    pipe, params
    """

    log.info('Preparing pipeline')

    combined_features = None
    if fsmethod1 is not None or fsmethod2 is not None:
        #feature selection pipeline
        fs1n = fsmethod1
        fs2n = fsmethod2

        #informing user
        info = 'Selecting features: FSMETHOD1: ' + fs1n
        if fs2n is not None:
            info += ', FSMETHOD2: ' + fs2n
        log.info(info)

        #union of feature selection processes
        fs1, fs1p = get_fsmethod(fs1n, n_feats, n_cpus, **fs1_kwargs)
        fs1p = append_to_keys(fs1p, fs1n + '__')
        if fs2n is not None:
            fs2, fs2p = get_fsmethod(fs2n, n_feats, n_cpus, **fs2_kwargs)
            fs2p = append_to_keys(fs2p, fs2n + '__')

            combined_features = FeatureUnion([(fs1n, fs1), (fs2n, fs2)])
            fsp = dict(list(fs1p.items()) + list(fs2p.items()))
        else:
            combined_features = FeatureUnion([(fs1n, fs1)])
            fsp = fs1p

    #classifier instance
    classif, clp = get_clfmethod(clfmethod, n_feats, **clf_kwargs)
    #clp     = append_to_keys(clgrid[clfmethod], clfmethod + '__')

    #if clfmethod == 'gmm':
    #    classif.means_ = np.array([X_train[y_train == i].mean(axis=0)
    #                     for i in xrange(n_class)])

    #creating pipeline
    if combined_features is not None:
        pipe = Pipeline([('fs', combined_features), ('cl', classif)])

        #arranging parameters for the whole pipeline
        clp = append_to_keys(clp, 'cl__')
        fsp = append_to_keys(fsp, 'fs__')
        params = dict(list(clp.items()) + list(fsp.items()))
    else:
        #pipe does not work
        #pipe = Pipeline([ ('cl', classif) ])
        #arranging parameters for the whole pipeline
        #clp = append_to_keys(clp, 'cl__')
        pipe = classif
        params = clp

    return pipe, params
