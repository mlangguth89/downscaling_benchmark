__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-01-20"
__update__ = "2022-01-22"


def provide_default(dict_in, keyname, default=None, required=False):
    """
    Returns values of key from input dictionary or alternatively its default

    :param dict_in: input dictionary
    :param keyname: name of key which should be added to dict_in if it is not already existing
    :param default: default value of key (returned if keyname is not present in dict_in)
    :param required: Forces existence of keyname in dict_in (otherwise, an error is returned)
    :return: value of requested key or its default retrieved from dict_in
    """

    if not required and default is None:
        raise ValueError("Provide default when existence of key in dictionary is not required.")

    if keyname not in dict_in.keys():
        if required:
            print(dict_in)
            raise ValueError("Could not find '{0}' in input dictionary.".format(keyname))
        return default
    else:
        return dict_in[keyname]