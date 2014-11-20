# -*- coding: utf-8 -*-
import os.path as op
import sys
import pytest
import sklearn

from darwin import instance


CWD = op.dirname(op.realpath(__file__))
MODULE_DIR = op.join(CWD, '..', 'darwin')


class TestImports(object):

    def test_import_this(self):
        """Test import_this function"""
        cls = instance.import_this('collections.defaultdict')
        assert(cls.__module__ == 'collections')

    def test_import_this_raises(self):
        """Test import_this function raising an Exception"""
        pytest.raises(Exception, instance.import_this, 'wrong.module')

    def test_import_pyfile_ioerror(self):
        pytest.raises(IOError, instance.import_pyfile, op.join(MODULE_DIR, 'dontexists'))

    def test_import_pyfile(self):
        imp_inst = instance.import_pyfile(op.join(MODULE_DIR, 'version.py'))
        #assert('imp_inst' in sys.modules)
        assert(hasattr(imp_inst, 'VERSION'))


class TestInstantiator(object):

    def test_learner_yaml_instance(self):
        inst = instance.MethodInstantiator(op.join(MODULE_DIR, 'learners.yml'))
        learner_item_name = 'LinearSVC'
        cls = inst.get_method_instance(learner_item_name)
        item = inst.get_yaml_item(learner_item_name)
        assert(type(cls).__name__ == item['class'].split('.')[-1])

    def test_learner_yaml_raises_ioerror(self):
        pytest.raises(IOError, instance.MethodInstantiator, 'notexist')

    def test_learner_yaml_raises_keyerror(self):
        inst = instance.MethodInstantiator(op.join(MODULE_DIR, 'learners.yml'))
        learner_item_name = 'NotExist'
        pytest.raises(KeyError, inst.get_method_instance, learner_item_name)


class TestLearnerInstantiator(object):

    def test_learner_yaml_instance(self):
        inst = instance.LearnerInstantiator()
        learner_item_name = 'LinearSVC'
        cls = inst.get_method_instance(learner_item_name)
        item = inst.get_yaml_item(learner_item_name)
        assert(type(cls).__name__ == item['class'].split('.')[-1])


class TestSelectorInstantiator(object):

    def test_selector_with_class_instance(self):
        selin = instance.SelectorInstantiator()
        selin.method_name = 'RFE'
        assert(isinstance(selin.default_params['estimator'], sklearn.svm.SVC))

    def test_selector_with_function(self):
        selin = instance.SelectorInstantiator()
        selin.method_name = 'SelectPercentile'
        assert(hasattr(selin.default_params['score_func'], '__call__'))
