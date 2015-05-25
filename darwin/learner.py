# -*- coding: utf-8 -*-

from .instance import LearnerInstantiator
from .utils.persist import PersistenceMixin


class Learner(PersistenceMixin):

        def __init__(self, learner_instance, param_grid=None):
            self.model      = learner_instance
            self.param_grid = param_grid

        @property
        def model_type(self):
            """A string representation of the underlying modeltype"""
            return type(self.model)
