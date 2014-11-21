# -*- coding: utf-8 -*-
import os.path as op
import pytest
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


def test_learner_yaml_instance():
    inst = instance.Instantiator(op.join(MODULE_DIR, 'learners.yml'))
    learner_item_name = 'LinearSVC'
    cls = inst.get_class_instance(learner_item_name)
    item = inst.get_yaml_item(learner_item_name)
    assert(type(cls).__name__ == item['class'].split('.')[-1])

def test_learner_yaml_raises_ioerror():
    pytest.raises(IOError, instance.Instantiator, 'notexist')

def test_learner_yaml_raises_keyerror():
    inst = instance.Instantiator(op.join(MODULE_DIR, 'learners.yml'))
    learner_item_name = 'NotExist'
    pytest.raises(KeyError, inst.get_class_instance, learner_item_name)

def test_all_default_learner_instances():
    inst = instance.LearnerInstantiator()
    for cls_name in inst.yaml
