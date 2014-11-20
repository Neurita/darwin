# -*- coding: utf-8 -*-
"""
YAML Class Instantiator
"""
import os.path as op
import sys
import yaml
import logging
import importlib

log = logging.getLogger(__name__)


def import_this(object_module_path):
    """Import any class or function to the global Python environment.j
       The module_path argument specifies what class to import in
       absolute or relative terms (e.g. either pkg.mod or ..mod).
       If the name is specified in relative terms, then the package argument
       must be set to the name of the package which is to act as the anchor
       for resolving the package name.

    Parameters
    ----------
    module_path: str
        Path to the module to be imported

    Returns
    -------
    The specified module will be inserted into sys.modules and returned.
    """
    try:
        mod_path_list = object_module_path.split('.')

        mod = import_module('.'.join(mod_path_list[:-1]))
        return getattr(mod, mod_path_list[-1])
    except:
        log.exception('Importing object {}.'.format(object_module_path))
        raise


def import_module(module_path):
    """Import any module to the global Python environment.
       The module_path argument specifies what module to import in
       absolute or relative terms (e.g. either pkg.mod or ..mod).
       If the name is specified in relative terms, then the package argument
       must be set to the name of the package which is to act as the anchor
       for resolving the package name (e.g. import_module('..mod', 'pkg.subpkg')

       will import pkg.mod).

    Parameters
    ----------
    module_path: str
        Path to the module to be imported

    Returns
    -------
    The specified module will be inserted into sys.modules and returned.
    """
    try:
        mod = importlib.import_module(module_path)
        return mod
    except:
        log.exception('Importing module {}.'.format(module_path))
        raise


def import_pyfile(filepath, mod_name=None):
    """
    Imports the contents of filepath as a Python module.

    Parameters
    ----------
    filepath: str
        Path to the .py file to be imported as a module

    mod_name: str
        Name of the module when imported

    Returns
    -------
    mod
        The imported module

    Raises
    ------
    IOError
        If file is not found
    """
    if not op.exists(filepath):
        msg = 'File {} not found.'.format(filepath)
        raise IOError(msg)

    if sys.version_info.major == 3:
        import importlib.machinery
        loader = importlib.machinery.SourceFileLoader('', filepath)
        mod = loader.load_module(mod_name)
    else:
        import imp
        mod = imp.load_source(mod_name, filepath)

    return mod


def instantiate_this(class_path, init_args):
    """Instantiates an object of the class in class_path with the given initializantion arguments.

    Parameters
    ----------
    class_path: str
        String to the path of the class.

    init_args: dict
        Dictionary of the names and values of the initialization arguments to the class

    Return
    ------
    Instantiated object
    """
    try:
        cls = import_this(class_path)
        return cls(*init_args)
    except:
        log.exception('Error instantiating class {} with the arguments {}.'.format(class_path, init_args))
        raise


class MethodInstantiator(object):

    """
    YAML Class Instantiator for classifiers and feature selections methods.
    For now, it only works on classes with the scikit-learn interface.

    Parameters
    ----------
    ymlpath: str
        Path to a YAML Path with the syntax of learners.yml and selectors.yml
    """

    def __init__(self, ymlpath=None):

        if ymlpath is not None:
            self._ymlpath = ymlpath

        self._method_name = None

        try:
            with open(self._ymlpath, 'rt') as f:
                self.yamldata = yaml.load(f)

        except IOError:
            log.exception("File {} not found.".format(ymlpath))
            raise
        except:
            log.exception("Error reading file {}.".format(ymlpath))
            raise

    def get_yaml_item(self, method_name):
        """Return the item in the YAML file corresponding to the given method_name

        Parameters
        ----------
        method_name: str

        Returns
        -------
        yml_item: dict
            The item in the YAML file corresponding to the given method_name

        Raises
        ------
        KeyError
            If the method_name is not found
        """
        try:
            return self.yamldata[method_name]
        except KeyError as ke:
            log.exception('Could not find item {}.'.format(method_name))
            raise

    def has_method(self):
        return self._method_name is not None

    @property
    def methods(self):
        return self.yamldata.keys()

    def _check_method_name(self):
        if not self.has_method:
            msg = 'The member method_name should be defined before accessing its properties.'
            log.error(msg)
            raise AttributeError(msg)

    @property
    def method_name(self):
        """The name of the method to be instantiated.
        It's definition should be in self.ymlpath.
        """
        return self._method_name

    @property
    def default_params(self):
        """The default construction parameters indicated by self.method_name"""
        self._check_method_name()
        return self.get_default_params(self._method_name)

    @property
    def param_grid(self):
        """The parameter grid indicated by self.method_name"""
        self._check_method_name()
        return self.get_param_grid(self._method_name)

    @property
    def instance(self):
        """The method instance indicated by self.method_name"""
        self._check_method_name()
        return self.get_learner_instance(self._method_name)

    @method_name.setter
    def method_name(self, method_name):
        self._method_name = method_name

    def get_default_params(self, method_name):
        """Import the needed module for the class and return the instance of a method
            defined in the yamlpath yml file.

        Parameters
        ----------
        method_name: str
            Name of the method in the ymlpath yaml file.

        Returns
        -------
        cls: class
            The class

        Raises
        ------
        KeyError
            If the method_name is not found

        ImportError
            If the there is any error importing the class
        """
        def get_if_any_instance(param_def):
            if not isinstance(param_def, dict):
                return None

            if 'class' in param_def:
                return instantiate_this(param_def['class'], param_def['default'])
            elif 'function' in param_def:
                return import_this(param_def['function'])

            return None

        try:
            class_data = self.get_yaml_item(method_name)
            def_parms = class_data['default']

            for parm_name in def_parms:
                obj = get_if_any_instance(def_parms[parm_name])
                if obj is not None:
                    def_parms[parm_name] = obj

            return def_parms
        except ImportError:
            log.exception("Error importing module class {}.".format(method_name))
            raise
        except:
            log.exception("Error reading definition for method {} in {}.".format(method_name, self._ymlpath))
            raise

    def get_method_instance(self, method_name):
        """Import the needed module for the class and return the instance of a method
            defined in the yamlpath yml file.

        Parameters
        ----------
        method_name: str
            Name of the method in the ymlpath yaml file.

        Returns
        -------
        cls: class
            The class

        Raises
        ------
        KeyError
            If the method_name is not found

        ImportError
            If the there is any error importing the class

        """
        try:
            class_data = self.get_yaml_item(method_name)
            return instantiate_this(class_data['class'], self.get_default_params(method_name))
        except ImportError:
            log.exception("Error importing module class {}.".format(method_name))
            raise
        except:
            log.exception("Error reading definition for method {} in {}.".format(method_name, self._ymlpath))
            raise

    def get_param_grid(self, method_name):
        """Return the defined parameter grid for the given learner class.

        Parameters
        ----------
        class_name: str

        Returns
        -------
        cls: class
            The class

        Raises
        ------
        KeyError
            If the method_name is not found

        ImportError
            If the there is any error importing the class
        """
        try:
            class_data = self.get_yaml_item(method_name)
            return class_data['param_grid']
        except ImportError:
            log.exception("Error importing module class {}.".format(method_name))
            raise
        except:
            log.exception("Error reading definition for method {} in {}.".format(method_name, self._ymlpath))
            raise

    def get_method_with_grid(self, method_name):
        """Calls get_method_instance and get_param_grid in one function.

        Parameters
        ----------
        method_name: str
            Name of the the method in learners.yml.
            See darwin/learners.yml for valid choices.
            The name of most of the scikit-learn Classification and Regression
            classes should work.

        Returns
        -------
        learner_instance: class of the learner
            The learner object ready to use.

        param_grid: dict
            Parameter grid
        """
        try:
            return self.get_method_instance(method_name), self.get_param_grid(method_name)
        except:
            log.exception("Error obtaining definition for method {} in {}.".format(method_name, self._ymlpath))
            raise


class LearnerInstantiator(MethodInstantiator):
    """
    Instantiator for the internal learners.yml file
    """
    _ymlpath = op.join(op.dirname(__file__), 'learners.yml')

    #def get_learner():
    #

class SelectorInstantiator(MethodInstantiator):
    """
    Instantiator for the internal selectors.yml file
    """
    _ymlpath = op.join(op.dirname(__file__), 'selectors.yml')
