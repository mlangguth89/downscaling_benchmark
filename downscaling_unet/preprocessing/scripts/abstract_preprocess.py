__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-03-16"
__update__ = "2022-03-16"

import os

class Abstract_Preprocessing(object):
    """
    Abstract class for preprocessing
    """
    def __init__(self, name_preprocess: str, source_dir: str, target_dir: str, *args, **kwargs):
        """
        Basic initialization.
        :param name_preprocess: name of preprocessing chain for easy identification
        """
        method = Abstract_Preprocessing.__init__.__name__
        # sanity check
        assert isinstance(name_preprocess, str), "%{0}: name_preprocess must be a string.".format(method)
        assert os.path.isdir(source_dir), "%{0}: Parsed source_dir '{1}' does not exist.".format(method, source_dir)

        self.name_preprocess = name_preprocess
        self.source_dir = source_dir
        self.target_dir = Abstract_Preprocessing.check_target_dir(target_dir)

    def __call__(self):
        """
        To be defined in child class.
        """
        method = Abstract_Preprocessing.__call__.__name__

        raise NotImplementedError("Method {0} is not implemented yet. Cannot continue.".format(method))

    @staticmethod
    def check_target_dir(target_dir):
        method = Abstract_Preprocessing.check_target_dir.__name__

        if not os.path.isdir(target_dir):
            try:
                print("%{0}: Create target directory '{1}'.".format(method, target_dir))
                os.makedirs(target_dir)
            except Exception as err:
                print("%{0}: Problem creating target directory '{1}'. Inspect raised error.".format(method, target_dir))
                raise err
        else:
            pass

        return target_dir
