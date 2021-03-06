# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
#Authors:
# Alexandre Manhaes Savio <alexsavio@gmail.com>
# Darya Chyzhyk <darya.chyzhyk@gmail.com>
# Borja Ayerdi <ayerdi.borja@gmail.com>
# Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
# Neurita S.L.
#
# BSD 3-Clause License
#
# 2014, Alexandre Manhaes Savio
# Use this at your own risk!
#------------------------------------------------------------------------------


import logging

import numpy                    as np
from   scipy                    import stats
from   collections              import OrderedDict
from   sklearn.grid_search      import GridSearchCV
from   sklearn.preprocessing    import StandardScaler
from   sklearn.cross_validation import LeaveOneOut

from   .utils.strings           import append_to_keys
from   .utils.printable         import Printable
from   .sklearn_utils           import (get_pipeline, get_cv_method)
from   .instance                import (LearnerInstantiator, SelectorInstantiator)
from   .results                 import (ClassificationResult, ClassificationMetrics,
                                        classification_metrics, get_cv_classification_metrics,
                                        enlist_cv_results_from_dict)

log = logging.getLogger(__name__)


#Classification Pipeline
class ClassificationPipeline(Printable):
    """This class wraps a classification pipeline with grid search.
    Here you can select up to two feature selection methods, one feature
    scaler method, one classifier. It is also possible to specify
    the cross-validation method and the grid search objective function.

    This class uses all functions that are in .sklearn_utils.py and
    .results.py. If you need more details on what choices you can use for
    each pipeline parameter, please have a look there.

    Parameters
    ----------

    clfmethod: str
        Name of the the method in learners.yml.
        See darwin/learners.yml for valid choices.
        The name of most of the scikit-learn Classification and Regression
        classes should work.

    n_feats: int
        Number of features of the input dataset. This is useful for
        adjusting the feature selection and classification grid search
        parameters.

    scaler: sklearn scaler object

    cvmethod  : string or int
        String with a number or number for K, for a K-fold method.
        'loo' for LeaveOneOut

    stratified: bool
        Indicates whether to use a Stratified K-fold approach (in case it is a K-fold what you choose for cvmethod)

    n_cpus: int
        Number of CPUS to be used in the Grid Search

    gs_scoring: str
        Grid search scoring objective function.
    """

    learner_instantiator  = LearnerInstantiator()
    selector_instantiator = SelectorInstantiator()

    def __init__(self, clfmethod, scaler=StandardScaler(), cvmethod='10',
                 stratified=True, n_cpus=1, gs_scoring='accuracy'):

        self.clfmethod  = clfmethod
        self.fsmethods  = []

        self._pipe      = None
        self._paramgrid = None
        self._cv        = None
        self._gs        = None
        self._results   = None
        self._metrics   = None

        self.cvmethod   = cvmethod
        self.stratified = stratified
        self.scaler     = scaler
        self.n_cpus     = n_cpus
        self.gs_scoring = gs_scoring

        self.reset()

    def _append_clsname_to_keys(self, adict, cls):
        if adict is None:
            return {}
        return append_to_keys(adict, cls.__name__)

    def add_feature_selection(self, fsmethod_name):
        """ Appends another feature selection method in self.fsmethods and rebuilds the pipeline calling self.reset().

        Parameters
        ----------
        fsmethod_name: str
            See get_fsmethod for possible choices
        """
        self.fsmethods.append(fsmethod_name)
        self.reset()

    def reset(self):
        """Remakes the pipeline and the gridsearch objects.

        You can use this to modify parameters of this object and this will call
         the necessary functions to remake the pipeline.
        """
        self._pipe      = None
        self._paramgrid = None
        self._cv        = None
        self._gs        = None
        self._results   = None
        self._metrics   = None

        try:
            fsmethods = []
            self._paramgrid = {}
            for fsmethod in self.fsmethods:
                fsm, fsm_params = self.selector_instantiator.get_method_with_grid(fsmethod)
                self._paramgrid.update(self._append_clsname_to_keys(fsm_params, type(fsm)))

            clfm, clfm_params = self.learner_instantiator.get_method_with_grid(self.clfmethod)
            if fsmethods:
                self._pipe = get_pipeline(fsmethods, clfm)
                self._paramgrid.update(self._append_clsname_to_keys(clfm_params, type(clfm)))
            else:
                self._pipe = clfm
                self._paramgrid = clfm_params

            #creating grid search
            self._gs = GridSearchCV(self._pipe, self._paramgrid, n_jobs=self.n_cpus, verbose=0, scoring=self.gs_scoring)
        except Exception as exc:
            log.exception('Error instantiating grid search. {}'.format(str(exc)))
            raise

    def cross_validation(self, samples, targets, cvmethod=None):
        """Performs a cross-validation against a dataset and its labels.

        Parameters
        ----------
        samples: array_like

        targets: vector or list
            Class labels set in the same order as in samples

        cv: sklearn.crossvalidation class

        Returns
        -------
        Classification_Results, Classification Metrics
        """
        if cvmethod is None:
            self._cv = get_cv_method(targets, self.cvmethod, self.stratified)
        else:
            self._cv = cvmethod

        self.n_feats = samples.shape[1]

        #We use dictionaries to save each fold classification result
        #because we will need to identify all sets of results to one fold.
        #If we used lists, we would loose track of folds if something went
        #wrong.
        preds      = OrderedDict()
        probs      = OrderedDict()
        truth      = OrderedDict()
        best_pars  = OrderedDict()
        importance = OrderedDict()

        fold_count = 0
        for train, test in self._cv:
            log.debug('Processing fold ' + str(fold_count))

            #data cv separation
            x_train, x_test, y_train, y_test = samples[train, :], samples[test, :], targets[train], targets[test]

            # We correct NaN values in x_train and x_test
            nan_mean  = stats.nanmean(x_train)
            nan_train = np.isnan(x_train)
            nan_test  = np.isnan(x_test)

            #remove Nan values
            x_test[nan_test] = 0
            x_test = x_test + nan_test*nan_mean

            x_train[nan_train] = 0
            x_train = x_train + nan_train*nan_mean

            #y_train = y_train.ravel()
            #y_test = y_test.ravel()

            #scaling
            #if clfmethod == 'linearsvc' or clfmethod == 'onevsonesvc':
            if self.scaler is not None:
                log.debug('Normalizing data with: {}'.format(str(self.scaler)))
                x_train = self.scaler.fit_transform(x_train)
                x_test  = self.scaler.transform(x_test)

            #do it
            log.debug('Running grid search for fold {}'.format(fold_count))
            self._gs.fit(x_train, y_train)

            log.debug('Predicting on test set')

            #predictions
            preds    [fold_count] = self._gs.predict(x_test)
            truth    [fold_count] = y_test
            best_pars[fold_count] = self._gs.best_params_

            #features importances
            if hasattr(self._gs.best_estimator_, 'support_vectors_'):
                imp = self._gs.best_estimator_.support_vectors_
            elif hasattr(self._gs.best_estimator_, 'feature_importances_'):
                imp = self._gs.best_estimator_.feature_importances_
            else:
                imp = None

            importance[fold_count] = imp

            #best grid-search parameters
            try:
                probs[fold_count] = self._gs.predict_proba(x_test)
            except:
                probs[fold_count] = None

            log.debug('Result: {} classifies as {}.'.format(y_test, preds[fold_count]))

            fold_count += 1

        #summarize results
        has_values = lambda adict: bool([i for i in adict if adict[i] is not None])

        if not has_values(probs):
            probs = None
        if not has_values(importance):
            importance = None

        if isinstance(self._cv, LeaveOneOut):
            truth, preds, probs, labels = enlist_cv_results_from_dict(truth, preds, probs)
        else:
            labels = np.unique(targets)

        self._results = ClassificationResult(preds, probs, truth, best_pars, self._cv, importance, targets, labels)

        #calculate performance metrics
        self._metrics = self.result_metrics()

        return self._results, self._metrics

    def result_metrics(self, classification_results=None, cvmethod=None):
        """Return the Accuracy, Sensitivity, Specificity, Precision, F1-Score
        and Area-under-ROC of given classification results or self._results
        if None.

        Parameters
        ----------
        classification_results: results.Classification_Result
            If None, will use the ones from self.

        cvmethod: sklearn.crossvalidation class
            CV method used to obtain the results.
            If None, will use the one from self.

        Returns
        -------
        If self.cvmethod is LeaveOneOut then return the average values for
        each measure cited above in a results.Classification_Metrics object.

        Otherwhise return two results.Classification_Metrics, the first is the
         average and the second, the standard deviations.
        """
        cr = classification_results

        if cr is None:
            if self._results is None:
                log.error('Cross-validation should be performed before this.')
                return None
            else:
                cr = self._results

        if cvmethod is None:
            cvmethod = self._cv

        if isinstance(cvmethod, LeaveOneOut):

            acc, sens, spec, \
            prec, f1, auc = classification_metrics(cr.cv_targets,
                                                   cr.predictions,
                                                   cr.probabilities,
                                                   cr.labels)

            return ClassificationMetrics(acc, sens, spec, prec, f1, auc)

        else:
            metrics = get_cv_classification_metrics(cr.cv_targets,
                                                    cr.predictions,
                                                    cr.probabilities)

            avg_metrics = metrics.mean(axis=0)
            std_metrics = metrics.std(axis=0)

            avgs = ClassificationMetrics(*tuple(avg_metrics))
            stds = ClassificationMetrics(*tuple(std_metrics))

            return avgs, stds
