
import numpy as np
from sklearn import svm, datasets

from darwin.pipeline import ClassificationPipeline


def test_binary_classification_with_classification_pipeline():
    # generate the dataset
    n_samples = 100
    n_features = 20
    x, y = datasets.make_gaussian_quantiles(mean=None, cov=1.0, n_samples=n_samples, n_features=n_features, n_classes=2,
                                            shuffle=True, random_state=1)

    # -- test with darwin
    classifier_name = 'RBFSVC' #'linsvm'
    cvmethod = '10'
    n_feats = x.shape[1]

    pipe = ClassificationPipeline(n_feats=n_feats, clfmethod=classifier_name, cvmethod=cvmethod)
    results, metrics = pipe.cross_validation(x, y)
    assert(results is not None)

    #return results, metrics

#results, metrics = test_binary_classification_with_classification_pipeline()
# def test_
#
#     inst = instance.LearnerInstantiator()
#     learner_item_name = 'LinearSVC'
#     classifier, param_grid = inst.get_method_with_grid(learner_item_name)
