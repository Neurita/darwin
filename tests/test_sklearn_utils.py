from unittest import TestCase

import darwin.sklearn_utils as dr
__author__ = 'oier'


class test_learners(TestCase):
    def test_get_clfmethod(self):
        met, gr = dr.get_clfmethod('DecisionTreeClassifier')
        assert( str(type(met)).find('DecisionTreeClassifier'))
        assert(gr['criterion'] == ['gini', 'entropy'] )

class test_selectors(TestCase):

    def test_get_selmethos(self):
        met, gr = dr.get_fsmethod('RFE')
        assert( str(type(met)).find('RFE'))
        assert(gr['step'] == [0.01, 0.05, 0.1])