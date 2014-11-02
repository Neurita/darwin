
import joblib
from .version import __version__, VERSION


class PersistenceMixin(object):

    """
    Mixin that adds joblib persistence load and save function to any class.
    """
    @classmethod
    def from_file(cls, objdump_path):
        '''
        Parameters
        ----------
        objdump_path: str
            Path to the object dump file.

        Returns
        -------
        instance
            New instance of an object from the pickle at the specified path.
        '''
        obj_version, object = joblib.load(objdump_path)
        # Check that we've actually loaded a PersistenceMixin (or sub-class)
        if not isinstance(object, cls):
            raise ValueError(('The pickle stored at {} does not contain ' +
                              'a {} object.').format(objdump_path, cls))
            # Check that versions are compatible. (Currently, this just checks
            # that major versions match)
        elif obj_version[0] == VERSION[0]:
            if not hasattr(object, 'sampler'):
                object.sampler = None
                return learner
            else:
                raise ValueError(("{} stored in pickle file {} was created with version {} "
                                  "of {}, which is incompatible with the current version "
                                  "{}").format(cls, objdump_path, __name__,
                                               '.'.join(obj_version), '.'.join(VERSION)))

    def load(self, objdump_path):
        '''Replace the current object instance with a saved object.

        Parameters
        ----------
        objdump_path: str
            The path to the file to load.
        '''
        del self.__dict__
        self.__dict__ = Learner.from_file(objdump_path).__dict__

    def save(self, objdump_path):
        '''Save the learner to a file.
        Parameters
        ----------
        objdump_path: str
            The path to where you want to save the learner.
        '''
        # create the directory if it doesn't exist
        learner_dir = os.path.dirname(objdump_path)
        if not os.path.exists(learner_dir):
            os.makedirs(learner_dir)

        # write out the files
        joblib.dump((VERSION, self), objdump_path)
